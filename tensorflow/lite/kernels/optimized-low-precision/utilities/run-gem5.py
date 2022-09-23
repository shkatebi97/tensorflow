#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path
import subprocess
import optparse
from os import listdir, getcwd, chdir, stat
from os.path import isfile, join, splitext, isdir, split, dirname, realpath
from time import time_ns
import os, tempfile
from multiprocessing import Pool as ThreadPool
from os import listdir, sched_getaffinity
import json

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

def get_output(arguments: list, cwd: str | object = None, env: dict = None) -> str:
    mcpat = subprocess.run(arguments, stdout=subprocess.PIPE, cwd=cwd, env=env)
    return mcpat.stdout.decode('utf-8')
run = get_output

def send(runner: str,
        runner_send_options: list,
        tool_path: str,
        tool_name: str,
        tool_destination: str,
    ) -> str:
    return run([
        runner,
        *runner_send_options,
        tool_path,
        join(tool_destination, tool_name)
    ]), join(tool_destination, tool_name)

def set_env(variables: dict, file):
    payload = []
    for var in variables.keys():
        payload.append("{}={}".format(var, variables[var] if type(variables[var]) == str else "{}".format(variables[var])))
    file.write("\n".join(payload).encode('utf-8'))
    file.flush()
    
def get_temp_file(delete: bool = False):
    return tempfile.NamedTemporaryFile(delete=delete)

def close_temp_file(tmp):
    tmp.close()
    os.unlink(tmp.name)

parser = optparse.OptionParser()


parser.add_option('-n', '--num-runs',
    action="store", dest="num_runs",
    help="The number of main runs", default=5)
parser.add_option('-N', '--num-warmup-runs',
    action="store", dest="num_warmup_runs",
    help="The number of warmup runs", default=1)
parser.add_option('-d', '--dont-run-benchmark',
    action="append", dest="dont_benchmark",
    help="Don't run benchmark", default=[])
parser.add_option('-o', '--output-dir',
    action="store", dest="output_dir",
    help="Set output directory", default=".")
parser.add_option('-m', '--model-dir',
    action="store", dest="model_dir",
    help="Set model directory", default=".")
parser.add_option('-l', '--single-model',
    action="store", dest="single_model",
    help="Enables single model experiment and will discard '--model-dir'", default=None)
parser.add_option('-L', '--single-fp32-model',
    action="store", dest="single_fp32_model",
    help="Enables single model experiment and will discard '--model-dir'", default=None)
parser.add_option('-P', '--select-models-match-pattern',
    action="store", dest="models_pattern",
    help="", default=None)
parser.add_option('-M', '--select-models-dont-match-pattern',
    action="store", dest="models_dont_pattern",
    help="", default=None)
parser.add_option('-b', '--single-benchmark',
    action="store", dest="single_benchmark",
    help="Enables single benchmark experiment and will uses '--output-dir' as main benchmark path", default=None)
parser.add_option('-e', '--add-env-variable',
    action="append", dest="env_variables",
    help="Add ENV variables", default=[])
parser.add_option('-E', '--add-env-variable-to-host',
    action="append", dest="env_variables_host",
    help="Add ENV variables to host", default=[])
parser.add_option('-r', '--repeat-model',
    action="append", dest="repeat_model",
    help="Will repeat the model", default=[])
parser.add_option('-p', '--use-gem5-fast',
    action="store_false", dest="gem5_prof",
    help="Set to use gem5.fast", default=True)
parser.add_option('-c', '--only-update-config',
    action="store_true", dest="only_update_config",
    help="Only update the configuration", default=False)
parser.add_option('-a', '--discard-cache',
    action="store_true", dest="discard_cache",
    help="Discards Cache", default=False)
parser.add_option('-s', '--shadow-run',
    action="store_true", dest="shadow_run",
    help="Enables shadow run, and disables ordinary run", default=False)
parser.add_option('-S', '--add-gem5-spec',
    action="append", dest="gem5_specs",
    help="Add a gem5 spec", default=[])
parser.add_option('-R', '--remove-gem5-spec',
    action="append", dest="remove_gem5_specs",
    help="Remove a gem5 spec", default=[])
parser.add_option('-D', '--deepspeech_mode',
    action="store_true", dest="deepspeech",
    help="Enable DeepSpeech mode", default=False)
parser.add_option('--resnet50_mode',
    action="store_true", dest="resnet50",
    help="Enable ResNet50 mode", default=False)
parser.add_option('-C', '--enable-special-benchmarks',
    action="store_true", dest="special_benchmarks",
    help="Enable Special Bechmarks", default=False)
parser.add_option('-g', '--gem5-path',
    action="store", dest="gem5_path",
    help="Enable Special Bechmarks", default=None)
parser.add_option('-G', '--gem5-script-path',
    action="store", dest="gem5_script_path",
    help="Enable Special Bechmarks", default=None)

options, args = parser.parse_args()

gem5_static_options=[
    "-n", "1",
    "--mem-size", "4GB",
    "--mem-type", "LPDDR3_1600_1x32",
    "--caches", "--l2cache",
    "--cpu-clock", "2.45GHz",
    "--l1d_size", "128kB",
    "--l1i_size", "128kB",
    "--cpu-type", "ex5_big",
    *options.gem5_specs,
]
for gem5_spec in options.remove_gem5_specs:
    gem5_static_options.remove(gem5_spec)

benchmarks = [
    'I8-I8',
    'I8-I4',
    'I4-I8',
    'I4-I4',
    'I8-Ternary',
    'Ternary-I8',
    'Ternary-Ternary',
    'I8-Binary',
    'Binary-I8',
    'Binary-Binary',
    'Binary-Binary-XOR',
]
special_benchmarks = [
    'XNNPack',
    'GEMMLOWP',
    'No-Caching',
    'FP32',
    'XNNPack-FP32',
    'Eigen',
    'No-Caching-FP32',
]

if options.special_benchmarks:
    benchmarks = [ *benchmarks, *special_benchmarks ]

if options.single_benchmark:
    benchmarks = [ options.single_benchmark ]

for no_benchmark in options.dont_benchmark:
    try:
        idx = benchmarks.index(no_benchmark)
        benchmarks.pop(idx)
    except ValueError:
        continue

print("Running these benchmarks:\n\t{}".format('\n\t'.join(benchmarks)))


benchmarks_fast_forward = {
    'I8-I8': 1844674407370955161,
    'I8-I4': 1844674407370955161,
    'I4-I8': 1844674407370955161,
    'I4-I4': 1844674407370955161,
    'I8-Ternary': 1844674407370955161,
    'Ternary-I8': 1844674407370955161,
    'Ternary-Ternary': 1844674407370955161,
    'I8-Binary': 1844674407370955161,
    'Binary-I8': 1844674407370955161,
    'Binary-Binary': 1844674407370955161,
    'Binary-Binary-XOR': 1844674407370955161,
    'XNNPack': 1844674407370955161,
    'FP32': 1844674407370955161,
    'No-Caching': 1844674407370955161,
    'No-Caching-FP32': 1844674407370955161,
    'XNNPack-FP32': 1844674407370955161,
    'GEMMLOWP': 1844674407370955161,
    'Eigen': 1844674407370955161,
}
iterations = 5
verbose_level = 2
use_shared_kernels = True
use_fused_kernel = False
operation_size = 1
num_layers = 3
do_adb = False
do_gem5 = True
nproc = len(sched_getaffinity(0))
n_threads = nproc
discard_cache = options.discard_cache
if do_adb:
    tools_dir = "build"
    tool_name = "multiplication-example"
    tool_path = join(tools_dir, tool_name)
    runner_path = get_output(["which", "adb"])[:-1]
    runner_options = ["shell"]
    runner_send_options = ["push"]
    config_dict = {
        "iterations": iterations,
        "verbose_level": verbose_level,
        "use_shared_kernels": use_shared_kernels,
        "use_fused_kernel": use_fused_kernel,
        "operation_size": operation_size,
        "num_layers": num_layers,
        "do_adb": do_adb,
        "do_gem5": do_gem5,
        "n_threads": n_threads,

        "tools_dir": tools_dir,
        "tool_name": tool_name,
        "tool_path": tool_path,
        "runner_path": runner_path,
        "runner_options": runner_options,
        "runner_send_options": runner_send_options,
    }
else:
    if options.resnet50:
        simulator_dir = "/home/user9886/Project/gem5-02-09-2022-stable/build/ARM"
        simulator_name = "gem5.opt"
        
        if options.gem5_path:
            simulator_path = options.gem5_path
        else:
            simulator_path = join(simulator_dir, simulator_name)

        simulation_scripts_dir = "/home/user9886/Project/gem5-02-09-2022-stable/configs/example"
        simulation_script_name = "se.py"

        if options.gem5_script_path:
            simulator_path = options.gem5_script_path
        else:
            simulation_script_path = join(simulation_scripts_dir, simulation_script_name)
    else:
        simulator_dir = "/home/user9886/Project/gem5/build/ARM"
        simulator_name = "gem5.prof" if options.gem5_prof else "gem5.fast"
        if options.gem5_path:
            simulator_path = options.gem5_path
        else:
            simulator_path = join(simulator_dir, simulator_name)
        simulation_scripts_dir = "/home/user9886/Project/gem5/configs/example"
        simulation_script_name = "se.py"
        if options.gem5_script_path:
            simulation_script_path = options.gem5_script_path
        else:
            simulation_script_path = join(simulation_scripts_dir, simulation_script_name)
    tools_dir = "/home/user9886/Project/Experiments/po2-multiplication-kernel/submodules/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark"
    tool_name = "benchmark_model"
    tool_path = join(tools_dir, tool_name)
    output_dir = options.output_dir
    model_dir = options.model_dir if not options.single_model else join("/", *options.single_model.split("/")[:-1])
    dumpreset_on_each_tick = 10000000
    num_runs = options.num_runs
    warmup_runs = options.num_warmup_runs
    min_secs = 0.000000001
    warmup_min_secs = 0.000000001
    host_env_var = dict(map(lambda x: ( x.split("=")[0], x.split("=")[1] ), options.env_variables_host))
    config_dict = {
        "verbose_level": verbose_level,
        "use_shared_kernels": use_shared_kernels,
        "use_fused_kernel": use_fused_kernel,
        "operation_size": operation_size,
        "num_layers": num_layers,
        "do_adb": do_adb,
        "do_gem5": do_gem5,
        "n_threads": n_threads,

        "simulator_dir": simulator_dir,
        "simulator_name": simulator_name,
        "simulator_path": simulator_path,
        "simulation_scripts_dir": simulation_scripts_dir,
        "simulation_script_name": simulation_script_name,
        "simulation_script_path": simulation_script_path,
        "tools_dir": tools_dir,
        "tool_name": tool_name,
        "tool_path": tool_path,
        "output_dir": output_dir,
        "model_dir": model_dir,
        "dumpreset_on_each_tick": dumpreset_on_each_tick,
        "num_runs": num_runs,
        "iterations": num_runs,
        "warmup_runs": warmup_runs,
        "warmup_iterations": warmup_runs,
        "min_secs": min_secs,
        "warmup_min_secs": warmup_min_secs,
    }
    if options.single_model:
        config_dict["single_model"] = options.single_model
    if options.env_variables and len(options.env_variables) > 0:
        config_dict["env_variables"] = " ".join(options.env_variables)
    if options.env_variables_host and len(options.env_variables_host) > 0:
        config_dict["env_variables_host"] = " ".join(options.env_variables_host)

if isfile(join(output_dir, "latest_run.config")):
    with open(join(output_dir, "latest_run.config")) as json_file:
        previous_config_json = json.load(json_file)
else:
    previous_config_json = ""

current_config_json = json.loads(json.dumps(config_dict))

if previous_config_json:
    previous_config_json_t = previous_config_json
    previous_config_json_t["output_dir"] = None
    previous_config_json_t["model_dir"] = None
    current_config_json_t = current_config_json
    current_config_json_t["output_dir"] = None
    current_config_json_t["model_dir"] = None
    if ordered(current_config_json_t) != ordered(previous_config_json_t):
        if not options.only_update_config:
            print(f"Will Discard cache due to changes in the latest run config file. {join(output_dir, 'latest_run.config')}")
        else:
            print(f"Config file changed. Updating {join(output_dir, 'latest_run.config')}.")
        discard_cache = True

if not options.single_benchmark and not options.shadow_run:
    if not isdir(output_dir):
        Path(output_dir).mkdir()
    with open(join(output_dir, "latest_run.config"), 'w') as json_file:
        json.dump(current_config_json, json_file, indent="\t")

if options.only_update_config:
    exit()

if len(benchmarks) and do_adb:
    print("Sending:", runner_path, runner_send_options, tool_path, tool_name, "/data/local/tmp/")
    _, tool_remote_path = send(
        runner_path,
        runner_send_options,
        tool_path,
        tool_name,
        "/data/local/tmp/",
    )
    print("Sent:", tool_remote_path)

if do_adb:
    for benchmark in benchmarks:
        output = run([
            runner_path,
            *runner_options,
            "VERBOSE_LEVEL={}".format(verbose_level),
            "USE_SHARED_KERNEL={}".format("TRUE" if use_shared_kernels else "FALSE"),
            "USE_FUSED_LOG={}".format("TRUE" if use_fused_kernel else "FALSE"),
            "OPERATION_SIZE={}".format(operation_size),
            "NUM_LAYER={}".format(num_layers),
            "taskset", "f0",
            tool_remote_path,
            benchmark,
            "{}".format(iterations),
        ])[:-1]
        print(output)
elif do_gem5:
    def workload(args):
        benchmark, (model, discard_cache) = args
        benchmark_name = benchmark
        model_name = splitext(model)[0].split('/')[-1]
        start_t = time_ns()
        env_file = get_temp_file()
        num_threads = 1
        is_xnn_benchmark = False
        is_fp32_benchmark = False
        is_non_ruy_benchmark = False
        is_disabled_gemv_benchmark = False
        is_non_caching = False
        is_special_benchmark = False
        if benchmark_name == "XNNPack":
            is_xnn_benchmark = True
        elif benchmark_name == "FP32":
            is_fp32_benchmark = True
        elif benchmark_name == "XNNPack-FP32":
            is_fp32_benchmark = True
            is_xnn_benchmark = True
        elif benchmark_name == "No-Caching":
            is_non_caching = True
        elif benchmark_name == "No-Caching-FP32":
            is_non_caching = True
            is_fp32_benchmark = True
        elif benchmark_name == "GEMMLOWP":
            is_non_ruy_benchmark = True
            is_disabled_gemv_benchmark = True
            is_non_caching = True
        elif benchmark_name == "Eigen":
            is_non_ruy_benchmark = True
            is_fp32_benchmark = True
            is_disabled_gemv_benchmark = True
            is_non_caching = True
        is_special_benchmark = is_xnn_benchmark or is_fp32_benchmark or is_non_ruy_benchmark or is_disabled_gemv_benchmark or is_non_caching
        if options.single_benchmark:
            benchmark = ""
            if is_fp32_benchmark:
                model_name.replace('i8i8', 'f32f32')
        try:
            if isfile(join(output_dir, benchmark, model_name, "run.log")) and not discard_cache:
                with open(join(output_dir, benchmark, model_name, "run.log")) as f:
                    lines = f.readlines()
                    if len(lines) > 4 and "Inference (avg): " in lines[-4]:
                        print("[{}]-[{}] Found in cache. Skipping.".format(benchmark_name, model_name))
                        return ""
            if options.shadow_run:
                print("[{}]-[{}] Will be running gem5 in normal mode. Output file: {}".format(benchmark_name, model_name, join(output_dir, benchmark, model_name, "run.log")))
                return ""
            envs = {
                "USE_ALTER_TIMING": "TRUE",
                "USING_GEM5": "TRUE",
                "SWITCH_CPU_GEM5": "MAINSTART",
                "DisableGEMV" : "TRUE" if is_disabled_gemv_benchmark else "FALSE",
                # "ForceCaching": "TRUE" if not is_xnn_benchmark and not is_non_caching else "FALSE",
                "LowPrecisionFC": "{}".format(benchmark_name if not is_special_benchmark else "I8-I8"),
                "DismissFilterQuantization": "TRUE",
                "DismissInputQuantization": "TRUE",
                "LowPrecisionMultiBatched": "TRUE",
                "LowPrecisionSingleBatched": "TRUE",
            }
            env_variables = dict(map(lambda ev: (ev.split("=")[0], ev.split("=")[1]), options.env_variables))
            for k, v in env_variables.items():
                envs[k] = v
            set_env(envs, env_file)
            print("[{}]-[{}] Running gem5. Output file: {}".format(benchmark_name, model_name, join(output_dir, benchmark, model_name, "run.log")))
            Path(join(output_dir, benchmark, model_name)).mkdir(exist_ok=True, parents=True)
            output = run([
                simulator_path,
                "--redirect-stdout",
                "--stdout-file", "run.log",
                "--stderr-file", "err.log",
                "--listener-mode", "off",
                "-d", f"../../{join(benchmark, model_name)}",
                simulation_script_path,
                *gem5_static_options,
                "--fast-forward", f"{benchmarks_fast_forward[benchmark_name]}",
                # "--dumpreset-each-n-ticks", "{}".format(dumpreset_on_each_tick),
                "--env", env_file.name,
                "-c", tool_path if not is_non_ruy_benchmark else f"{tool_path}_non_ruy",
                "-o", "--graph={} --use_xnnpack={} --use_caching={} --num_threads={} --num_runs={} --warmup_runs={} --min_secs={} --warmup_min_secs={}".format(
                    join(model_dir.replace('i8i8', 'f32f32' if is_fp32_benchmark and not options.single_benchmark and not options.single_model else 'i8i8'), model if not options.single_model or not options.single_fp32_model or not is_fp32_benchmark else options.single_fp32_model.split("/")[-1]), 
                    "true" if is_xnn_benchmark else "false", 
                    "false" if is_non_caching else "true", 
                    num_threads, num_runs, warmup_runs, min_secs, warmup_min_secs, 
                ),
            ], cwd=join(output_dir, benchmark, model_name), env=host_env_var)
            end_t = time_ns()
            print("[{}]-[{}] Done in {:.2f} seconds.".format(benchmark_name, model_name, (end_t - start_t)/1e+9))
        finally:
            close_temp_file(env_file)
        return output
    models = []
    if options.single_model or options.model_dir:
        if options.single_model:
            model_name = options.single_model.split("/")[-1]
            models = [ model_name ]
        else:
            models = listdir(model_dir)
        models = list(filter(lambda model: isfile(join(model_dir, model)) and splitext(model)[1] == ".tflite", models))
        models = list(map(lambda model: (model, discard_cache), models))
        if options.models_pattern is not None:
            models = list(filter(lambda x: options.models_pattern in x[0], models))
        if options.models_dont_pattern is not None:
            models = list(filter(lambda x: options.models_dont_pattern not in x[0], models))
    if options.repeat_model and len(options.repeat_model) > 0:
        models.extend(list(map(lambda model: (model, True), options.repeat_model)))
    workloads_args = [ (benchmark, model) for benchmark in benchmarks for model in models ]

    print("Running with {} workers".format(min(len(models) * len(benchmarks), n_threads)))
    pool = ThreadPool(n_threads)
    cwd = getcwd()
    outputs = pool.map(workload, workloads_args, 1)
    chdir(cwd)

print("Generating output files for each benchmark.")
try:
    for i, benchmark in enumerate(benchmarks):
        benchmark_name = deepcopy(benchmark)
        if options.single_benchmark and not options.deepspeech and not options.resnet50:
            benchmark = ""
        if options.deepspeech:
            with open(join(output_dir, benchmark, "output-{}-{}.log".format(num_runs, warmup_runs)), 'w') as output_file:
                for j, (model, _) in enumerate(models):
                    if "model-" in splitext(model)[0].split('/')[-1]:
                        model_name = splitext(model)[0].split('/')[-1].split("model-")[1]
                    else:
                        model_name = splitext(model)[0].split('/')[-1]
                    output_file.write(model_name + "\n")
                    print("\t[{}] Processing {}".format(f"{i * len(models) + j}".center(f"{len(models) * len(benchmarks)}".__len__() + 2), join(output_dir, benchmark, model_name, "run.log")))
                    if isfile(join(output_dir, benchmark, model_name, "run.log")):
                        with open(join(output_dir, benchmark, model_name, "run.log")) as log_file:
                            lines = log_file.readlines()
                            if len(lines) > 4 and "Inference (avg): " in lines[-4]:
                                output_file.write(str(float(lines[-4].split("Inference (avg): ")[1][:-1])))
                            output_file.write("\n")
                    else:
                        print("No such file")
                        print(stat(join(output_dir, benchmark, model_name, "run.log")))
                        exit(-1)
        elif options.resnet50:
            with open(join(output_dir, benchmark, "output-{}-{}.log".format(num_runs, warmup_runs)), 'w') as output_file:
                for j, (model, _) in enumerate(models):
                    if "resnet50-" in splitext(model)[0].split('/')[-1]:
                        model_name = splitext(model)[0].split('/')[-1].split("resnet50-")[1]
                    else:
                        model_name = splitext(model)[0].split('/')[-1]
                    output_file.write(model_name + "\n")
                    print("\t[{}] Processing {}".format(f"{i * len(models) + j}".center(f"{len(models) * len(benchmarks)}".__len__() + 2), join(output_dir, benchmark, model_name, "run.log")))
                    if isfile(join(output_dir, benchmark, model_name, "run.log")):
                        with open(join(output_dir, benchmark, model_name, "run.log")) as log_file:
                            lines = log_file.readlines()
                            if len(lines) > 4 and "Inference (avg): " in lines[-4]:
                                output_file.write(str(float(lines[-4].split("Inference (avg): ")[1][:-1])))
                            output_file.write("\n")
                    else:
                        print("No such file")
                        print(stat(join(output_dir, benchmark, model_name, "run.log")))
                        exit(-1)
        else:
            with open(join(output_dir, benchmark, "output-{}-{}.log".format(num_runs, warmup_runs)), 'w') as output_file:
                for j, (model, _) in enumerate(models):
                    if "model-" in splitext(model)[0].split('/')[-1]:
                        model_name = splitext(model)[0].split('/')[-1].split("model-")[1]
                    else:
                        model_name = splitext(model)[0].split('/')[-1]
                    output_file.write(model_name + "\n")
                    print("\t[{}] Processing {}".format(f"{i * len(models) + j}".center(f"{len(models) * len(benchmarks)}".__len__() + 2), join(output_dir, benchmark, "model-" + model_name, "run.log")))
                    if isfile(join(output_dir, benchmark, "model-" + model_name, "run.log")):
                        with open(join(output_dir, benchmark, "model-" + model_name, "run.log")) as log_file:
                            lines = log_file.readlines()
                            if len(lines) > 4 and "Inference (avg): " in lines[-4]:
                                output_file.write(str(float(lines[-4].split("Inference (avg): ")[1][:-1])))
                            output_file.write("\n")
                    else:
                        print("No such file")
                        print(stat(join(output_dir, benchmark, "model-" + model_name, "run.log")))
                        exit(-1)
except FileNotFoundError as e:
    if not options.shadow_run:
        raise e

        