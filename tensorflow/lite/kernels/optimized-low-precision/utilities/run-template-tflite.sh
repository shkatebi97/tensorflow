#!/usr/bin/env bash
TIMEFORMAT='%3R'
function list_contains() {
        local list=("$@")
        local value=${list[-1]}
        unset list[-1]
        for v in ${list[@]}; do
                if [[ "$v" = "$value" ]]; then
                        return 1
                fi
        done
        return 0
}

function print_help {
        echo -e "$0 [options] [models_root_directory]"
        echo -e "OPTIONS:"
        echo -e "\t-n\t--num-iterations"
        echo -e "\t\tSet number of iterations.\n"
        echo -e "\t-d\t--models-dir"
        echo -e "\t\tSet remote models dir.\n"
        echo -e "\t-m\t--add-method"
        echo -e "\t\tAdd a method to methods list.\n"
        echo -e "\t-r\t--record-report-stat"
        echo -e "\t\tSet the path directory to stroe report and stat results.\n"
        echo -e "\t-p\t--disable-perf"
        echo -e "\t\tDisables running perf. Must be passed after '--record-report-stat'\n"
        echo -e "\t-h\t--help"
        echo -e "\t\tPrint this help.\n"
}

function exit_on_fail {
        eval $@
        if [ ! $? -eq 0 ]; then
                ret=$?
                echo "[x] Failed"
                adb shell settings put global stay_on_while_plugged_in 0
                exit
        fi
}

models_list=( \
    model-16-batch-10x32.tflite \
    model-16-batch-10x64.tflite \
    model-16-batch-10x128.tflite \
    model-16-batch-10x512.tflite \
    model-32-batch-10x32.tflite \
    model-32-batch-10x64.tflite \
    model-32-batch-10x128.tflite \
    model-32-batch-10x512.tflite \
    model-64-batch-10x32.tflite \
    model-64-batch-10x64.tflite \
    model-64-batch-10x128.tflite \
    model-64-batch-10x512.tflite \
    model-512-batch-10x512.tflite \
)

method_list=( \
    I8-I8 \
    I8-I4 \
    I4-I8 \
    I4-I4 \
    I8-Ternary \
    Ternary-I8 \
    Ternary-Ternary \
    I8-Binary \
    Binary-I8 \
    Binary-Binary \
)
iterations=200
warmup_iterations=1
min_secs=1
warmup_min_secs=0.5
models_dir=/data/local/tmp/test-models/different-sizes
fp32_models_dir=/data/local/tmp/test-models/different-sizes
use_simpleperf=0
report_path=./reports
models_root=.
do_sleep=0
adb_options=()
device_name=$(adb devices | head -2 | tail -1 | cut -f 1)
POSITIONAL=()
SLEEP_TIME_COEFFICIENT=5
CHECK_CACHE=1
single_model=
single_model_fp32=
taskset_mode=f0
taskset_modes=()
taskset_mode_added=0
model_runner="benchmark_model.0.1"
model_runner_no_ruy="benchmark_model_non_ruy.0.1"

XNNPACK_W8A8=0
TFLITE_W8A8=0
GEMMLOWP=0

RUY_FP32=0
XNNPACK_FP32=0
TFLITE_FP32=0
EIGEN=0

ULPPACK1=0
ULPPACK2=0
ULPPACK3=0
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -n|--num-iterations)
        iterations=$2
        shift
        shift
        ;;
    --num-warmup-iterations)
        warmup_iterations=$2
        shift
        shift
        ;;
    --min-secs)
        min_secs=$2
        shift
        shift
        ;;
    --warmup-min-secs)
        warmup_min_secs=$2
        shift
        shift
        ;;
    -d|--models-dir)
        models_dir=$2
        shift
        shift
        ;;
    --fp32-models-dir)
        fp32_models_dir=$2
        shift
        shift
        ;;
    --model-runner)
        model_runner=$2
        shift
        shift
        ;;
    --model-runner-no-ruy)
        model_runner_no_ruy=$2
        shift
        shift
        ;;
    -m|--add-method)
        method_list+=($2)
        shift
        shift
        ;;
    -c|--clear-methods)
        method_list=()
        shift
        ;;
    -g|--single-model)
        single_model=$2
        shift
        shift
        ;;
    --single-model-fp32)
        single_model_fp32=$2
        shift
        shift
        ;;
    -r|--record-report-stat)
        use_simpleperf=1
        report_path=$2
        shift
        shift
        ;;
    -p|--disbale-perf)
        use_simpleperf=0
        shift
        ;;
    -a|--add-adb-option)
        adb_options+=($2)
        shift
        shift
        ;;
    -i|--device-name)
        device_name=$2
        shift
        shift
        ;;
    -f|--sleep-multiply-coeff)
        SLEEP_TIME_COEFFICIENT=$2
        shift
        shift
        ;;
    -k|--add-taskset-mode)
        taskset_mode_added=1
        taskset_modes+=($2)
        shift
        shift
        ;;
    -s|--sleep-between-runs)
        do_sleep=1
        shift
        ;;
    -x|--discard-cache)
        CHECK_CACHE=0
        shift
        ;;
    --enable_XNNPACK_W8A8)
        XNNPACK_W8A8=1
        shift
        ;;
    --enable_TFLITE_W8A8)
        TFLITE_W8A8=1
        shift
        ;;
    --enable_GEMMLOWP)
        GEMMLOWP=1
        shift
        ;;
    --enable_RUY_FP32)
        RUY_FP32=1
        shift
        ;;
    --enable_XNNPACK_FP32)
        XNNPACK_FP32=1
        shift
        ;;
    --enable_TFLITE_FP32)
        TFLITE_FP32=1
        shift
        ;;
    --enable_EIGEN)
        EIGEN=1
        shift
        ;;
    --enable_ULPPACK1)
        ULPPACK1=1
        shift
        ;;
    --enable_ULPPACK2)
        ULPPACK2=1
        shift
        ;;
    --enable_ULPPACK3)
        ULPPACK3=1
        shift
        ;;
    -h|--help)
        print_help $@
        exit
        ;;
    *)
        POSITIONAL+=($1)
        shift
        ;;
  esac
done

set -- "${POSITIONAL[@]}"

if [[ ! -z $1 ]]; then
    echo "Setting models path to $1"
    models_root=$1
fi

if [[ $taskset_mode_added = 0 ]]; then
    taskset_modes=( $taskset_mode )
fi

adb ${adb_options[@]} shell settings put global stay_on_while_plugged_in 2

echo "Checking config file"
echo "{" > /tmp/latest_run.config
echo -e "\t\"iterations\": $iterations," >> /tmp/latest_run.config
echo -e "\t\"warmup_iterations\": $warmup_iterations," >> /tmp/latest_run.config
echo -e "\t\"min_secs\": $min_secs," >> /tmp/latest_run.config
echo -e "\t\"warmup_min_secs\": $warmup_min_secs," >> /tmp/latest_run.config
echo -e "\t\"models_dir\": \"$models_dir\"," >> /tmp/latest_run.config
echo -e "\t\"use_simpleperf\": $use_simpleperf," >> /tmp/latest_run.config
echo -e "\t\"report_path\": \"$report_path\"," >> /tmp/latest_run.config
echo -e "\t\"models_root\": \"$PWD/$models_root\"," >> /tmp/latest_run.config
echo -e "\t\"do_sleep\": $do_sleep," >> /tmp/latest_run.config
echo -e "\t\"adb_options\": \"$adb_options\"," >> /tmp/latest_run.config
echo -e "\t\"device_name\": \"$device_name\"," >> /tmp/latest_run.config
echo -e "\t\"SLEEP_TIME_COEFFICIENT\": $SLEEP_TIME_COEFFICIENT," >> /tmp/latest_run.config
echo -e "\t\"CHECK_CACHE\": $CHECK_CACHE," >> /tmp/latest_run.config
echo -e "\t\"single_model\": \"$single_model\"," >> /tmp/latest_run.config
echo -e "\t\"single_model_fp32\": \"$single_model_fp32\"," >> /tmp/latest_run.config
echo -e "\t\"taskset_modes\": \"${taskset_modes[@]}\"" >> /tmp/latest_run.config
echo "}" >> /tmp/latest_run.config

if ! cmp --silent -- $report_path/latest_run.config /tmp/latest_run.config; then
    echo [!] Config changed from lastest run, discarding cache...
    # CHECK_CACHE=0
fi

cp /tmp/latest_run.config $report_path/latest_run.config

for method in "${method_list[@]}"; do
    if [[ ! -d $report_path/$method ]]; then
        mkdir -p $report_path/$method
    fi
    if [[ $CHECK_CACHE = 0 && -f $report_path/$method/output-$iterations-$warmup_iterations.log ]]; then
        rm $report_path/$method/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/$method/output-$iterations-$warmup_iterations.log ]]; then
        touch $report_path/$method/output-$iterations-$warmup_iterations.log
    fi
done
if [[ $XNNPACK_W8A8 = 1 ]]; then
    if [[ ! -d $report_path/XNNPack ]]; then 
        mkdir -p $report_path/XNNPack
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/XNNPack/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/XNNPack/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/XNNPack/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/XNNPack/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $TFLITE_W8A8 = 1 ]]; then
    if [[ ! -d $report_path/No-Caching ]]; then 
        mkdir -p $report_path/No-Caching
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/No-Caching/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/No-Caching/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/No-Caching/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/No-Caching/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $GEMMLOWP = 1 ]]; then
    if [[ ! -d $report_path/GEMMLOWP ]]; then 
        mkdir -p $report_path/GEMMLOWP
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/GEMMLOWP/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/GEMMLOWP/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/GEMMLOWP/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/GEMMLOWP/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $RUY_FP32 = 1 ]]; then
    if [[ ! -d $report_path/FP32 ]]; then 
        mkdir -p $report_path/FP32
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/FP32/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/FP32/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/FP32/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/FP32/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $XNNPACK_FP32 = 1 ]]; then
    if [[ ! -d $report_path/XNNPack-FP32 ]]; then 
        mkdir -p $report_path/XNNPack-FP32
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/XNNPack-FP32/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/XNNPack-FP32/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/XNNPack-FP32/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/XNNPack-FP32/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $TFLITE_FP32 = 1 ]]; then
    if [[ ! -d $report_path/No-Caching-FP32 ]]; then 
        mkdir -p $report_path/No-Caching-FP32
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/No-Caching-FP32/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/No-Caching-FP32/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/No-Caching-FP32/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/No-Caching-FP32/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $EIGEN = 1 ]]; then
    if [[ ! -d $report_path/Eigen ]]; then 
        mkdir -p $report_path/Eigen
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/Eigen/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/Eigen/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/Eigen/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/Eigen/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $ULPPACK1 = 1 ]]; then
    if [[ ! -d $report_path/ULPPACK-W1A1 ]]; then 
        mkdir -p $report_path/ULPPACK-W1A1
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/ULPPACK-W1A1/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/ULPPACK-W1A1/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/ULPPACK-W1A1/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/ULPPACK-W1A1/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $ULPPACK2 = 1 ]]; then
    if [[ ! -d $report_path/ULPPACK-W2A2 ]]; then 
        mkdir -p $report_path/ULPPACK-W2A2
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/ULPPACK-W2A2/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/ULPPACK-W2A2/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/ULPPACK-W2A2/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/ULPPACK-W2A2/output-$iterations-$warmpup_iterations.log
    fi
fi
if [[ $ULPPACK3 = 1 ]]; then
    if [[ ! -d $report_path/ULPPACK-W3A3 ]]; then 
        mkdir -p $report_path/ULPPACK-W3A3
    fi
    if [[ $CEHCK_CACHE = 0 && -f $report_path/ULPPACK-W3A3/output-$iterations-$warmup_iterations.log ]]; then 
        rm $report_path/ULPPACK-W3A3/output-$iterations-$warmup_iterations.log
    fi
    if [[ ! -f $report_path/ULPPACK-W3A3/output-$iterations-$warmpup_iterations.log ]]; then 
        touch $report_path/ULPPACK-W3A3/output-$iterations-$warmpup_iterations.log
    fi
fi

echo "[!] Starting..."
for model in $models_root/*.tflite; do
    model_file=$(basename "$model")
    model_name=$(basename "$model" .tflite)
    model_name=${model_name#"model-"}
    if [[ ! -z $single_model && $model_name != $single_model ]]; then
        echo Skipping $model_name.
        continue
    fi
    model_path=$models_dir/$model_file
    if [[ ! -z $single_model ]]; then
        model_fp32_path=$models_dir/$single_model_fp32.tflite
    else
        model_fp32_path=$fp32_models_dir/$model_file
    fi
    echo "[!] Running model $model_name for $iterations iterations and $warmup_iterations warmup iterations."
    for method in "${method_list[@]}"; do
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -v LowPrecisionFC=$method \
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_path \
                                            --enable_op_profiling=true --use_caching=true \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                            LowPrecisionFC=$method \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_path \
                                        --enable_op_profiling=true --use_caching=true \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                    LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_path \
                                --enable_op_profiling=true --use_caching=true \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    done
    if [[ $XNNPACK_W8A8 = 1 ]]; then
        method="XNNPack"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_path \
                                            --enable_op_profiling=true --use_caching=true \
                                            --use_xnnpack=true --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs } \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_path \
                                        --enable_op_profiling=true --use_caching=true \
                                        --use_xnnpack=true --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_path \
                                --enable_op_profiling=true --use_caching=true \
                                --use_xnnpack=true --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $TFLITE_W8A8 = 1 ]]; then
        method="No-Caching"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_path \
                                            --enable_op_profiling=true --use_caching=false \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_path \
                                        --enable_op_profiling=true --use_caching=false \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_path \
                                --enable_op_profiling=true --use_caching=false \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $GEMMLOWP = 1 ]]; then
        method="GEMMLOWP"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -v LowPrecisionFC=$method -v LowPrecisionMultiBatched=FALSE  -v ForceCaching=TRUE\
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner_no_ruy" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner_no_ruy \
                                            --graph=$model_path \
                                            --enable_op_profiling=true \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                            LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner_no_ruy \
                                        --graph=$model_path \
                                        --enable_op_profiling=true \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                    LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner_no_ruy \
                                --graph=$model_path \
                                --enable_op_profiling=true \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $RUY_FP32 = 1 ]]; then
        method="FP32"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -v LowPrecisionFC=$method -v LowPrecisionMultiBatched=FALSE  -v ForceCaching=TRUE\
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_fp32_path \
                                            --enable_op_profiling=true --use_caching=true \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                            LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_fp32_path \
                                        --enable_op_profiling=true --use_caching=true \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                    LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_fp32_path \
                                --enable_op_profiling=true --use_caching=true \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $XNNPACK_FP32 = 1 ]]; then
        method="XNNPack-FP32"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_fp32_path \
                                            --enable_op_profiling=true \
                                            --use_xnnpack=true --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_fp32_path \
                                        --enable_op_profiling=true \
                                        --use_xnnpack=true --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_fp32_path \
                                --enable_op_profiling=true \
                                --use_xnnpack=true --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $TFLITE_FP32 = 1 ]]; then
        method="No-Caching-FP32"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_fp32_path \
                                            --enable_op_profiling=true --use_caching=false \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_fp32_path \
                                        --enable_op_profiling=true --use_caching=false \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_fp32_path \
                                --enable_op_profiling=true --use_caching=false \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $EIGEN = 1 ]]; then
        method="Eigen"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_fp32_path \
                                            --enable_op_profiling=true --use_caching=false \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_fp32_path \
                                        --enable_op_profiling=true --use_caching=false \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_fp32_path \
                                --enable_op_profiling=true --use_caching=false \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $ULPPACK1 = 1 ]]; then
        method="ULPPACK-W1A1"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -v LowPrecisionFC=$method -v LowPrecisionMultiBatched=FALSE  -v ForceCaching=TRUE\
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_path \
                                            --enable_op_profiling=true --use_caching=false \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                            LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_path \
                                        --enable_op_profiling=true --use_caching=false \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                    LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_path \
                                --enable_op_profiling=true --use_caching=false \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $ULPPACK2 = 1 ]]; then
        method="ULPPACK-W2A2"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -v LowPrecisionFC=$method -v LowPrecisionMultiBatched=FALSE  -v ForceCaching=TRUE\
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_path \
                                            --enable_op_profiling=true --use_caching=false \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                            LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_path \
                                        --enable_op_profiling=true --use_caching=false \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                    LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_path \
                                --enable_op_profiling=true --use_caching=false \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
    if [[ $ULPPACK3 = 1 ]]; then
        method="ULPPACK-W3A3"
        if [[ \
            ! -f $report_path/$method/output-$iterations-$warmup_iterations.log || \
            (\
                -z `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1` && \
                `grep $model_name $report_path/$method/output-$iterations-$warmup_iterations.log -A 1 | tail -1 | grep avg | cut -d '=' -f 1` != "avg" \
            ) || \
            ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || \
            `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || \
            $CHECK_CACHE = 0 \
        ]]; then
            echo $model_name >> $report_path/$method/output-$iterations-$warmup_iterations.log
            EXECTIME=0
            if [[ $use_simpleperf = 1 ]]; then
                echo -n "Running For Method $method (Detailed): "
                for taskset_mode in "${taskset_modes[@]}"; do
                    if [[ ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report || ! -f $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv || `wc -l $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report | cut -d ' ' -f 1` -lt 50 || $CHECK_CACHE = 0 ]]; then
                        EXECTIME=$({ TIMEFORMAT=%E; time ./adb-record-report.sh \
                            --device-name $device_name \
                            -g cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            --report-type caller \
                            --generate-stat-csv $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.csv \
                                                cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads \
                            -v LowPrecisionFC=$method -v LowPrecisionMultiBatched=FALSE  -v ForceCaching=TRUE\
                            -o $report_path/$method/simpleperf-$iterations-$warmup_iterations-$model_name.report \
                            --report-flags "--dsos /data/local/tmp/$model_runner" \
                                -- taskset $taskset_mode \
                                        /data/local/tmp/$model_runner \
                                            --graph=$model_path \
                                            --enable_op_profiling=true --use_caching=false \
                                            --use_xnnpack=false --num_threads=1 \
                                            --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                            --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                            2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                            | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    else
                        echo -n "(Found perf results) "
                        EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                            LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                                taskset $taskset_mode \
                                    /data/local/tmp/$model_runner \
                                        --graph=$model_path \
                                        --enable_op_profiling=true --use_caching=false \
                                        --use_xnnpack=false --num_threads=1 \
                                        --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                        --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                        2> /dev/null | grep "Inference (avg):" | tail -1 | tr -s ' ' \
                                        | cut -d ' ' -f 15 >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
                    fi
                    if [[ `tail -1 $report_path/$method/output-$iterations-$warmup_iterations.log` != $model_name ]]; then 
                        break
                    fi
                done
            else
                echo -n "Running For Method $method (Simple): "
                EXECTIME=$({ TIMEFORMAT=%E; time adb -s $device_name ${adb_options[@]} shell \
                    LowPrecisionFC=$method LowPrecisionMultiBatched=FALSE ForceCaching=TRUE \
                        taskset $taskset_mode \
                            /data/local/tmp/$model_runner \
                                --graph=$model_path \
                                --enable_op_profiling=true --use_caching=false \
                                --use_xnnpack=false --num_threads=1 \
                                --num_runs=$iterations --warmup_runs=$warmup_iterations \
                                --min_secs=$min_secs --warmup_min_secs=$warmup_min_secs \
                                2> /dev/null >> $report_path/$method/output-$iterations-$warmup_iterations.log; } 2>&1)
            fi
            SLEEP_TIME=$(echo "$EXECTIME * $SLEEP_TIME_COEFFICIENT" | bc)
            if [[ $do_sleep = 1 ]]; then
                echo -n "Sleeping... "
                sleep $SLEEP_TIME
                echo "Done."
            fi
        else
            echo "Running For Method $method: (Found in cache)"
        fi
    fi
done
adb ${adb_options[@]} shell settings put global stay_on_while_plugged_in 0