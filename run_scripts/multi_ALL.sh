WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=0
exp_title="fp16-lvlm-7b-single-M-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
# DRAFTING
draftings="MultimodalDraft"
eval_drafting="multimodal"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for multi_turn_task in $multi_turn_tasks
                do
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-single-T-multi-turn-999999"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
draftings="TextOnlyDraft"
eval_drafting="text-only"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for multi_turn_task in $multi_turn_tasks
                do
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MT-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm" # align with drfs, tgts
# DRAFTING
draftings="TabedMT"
eval_drafting="EvalTabedMT" # align with draftings
tabed_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')
python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MT-history-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="TabedMT"
eval_drafting="EvalTabedMT" # align with draftings
tabed_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for temperature_drafting_weight in $temperature_drafting_weights
                        do
                            for history_item in $history_items
                            do
                                for history_window in $history_windows
                                do
                                    for history_filter_top1_match in $history_filter_top1_matchs
                                    do
                                        for history_w_grid_measure in $history_w_grid_measures
                                        do
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-single-C-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm" # align with drfs, tgts
# DRAFTING
draftings="CaptionDraft"
eval_drafting="caption" # align with draftings
caption_types="<CAPTION>" # caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for caption_type in $caption_types
                do
                    for captioning_model in $captioning_models
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                caption_type=$caption_type captioning_model=$captioning_model \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-single-P-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="InferencePoolDraft"
eval_drafting="image-pool"
target_dim_image_poolings="144"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for target_dim_image_pooling in $target_dim_image_poolings
                do
                    for multi_turn_task in $multi_turn_tasks
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            target_dim_image_pooling=$target_dim_image_pooling \
                            multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MTC-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="TabedMTC"
eval_drafting="EvalTabedMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
tabed_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=False
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                confidence_type=$confidence_type \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MTP-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="TabedMTP"
eval_drafting="EvalTabedMTP" # align with draftings
tabed_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=False
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for temperature_drafting_weight in $temperature_drafting_weights
do
    for drf in $drfs
    do
        for tgt in $tgts
        do
            for dataset in $WholeData
            do
                for drafting in $draftings
                do
                    for tabed_rule in $tabed_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                for multi_turn_task in $multi_turn_tasks
                                do
                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                        drf=$drf tgt=$tgt \
                                        $drafting \
                                        tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                        target_dim_image_pooling=$target_dim_image_pooling \
                                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"

device_num=0
exp_title="fp16-lvlm-7b-IbED-MTC-history-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="TabedMTC"
eval_drafting="EvalTabedMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
tabed_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for temperature_drafting_weight in $temperature_drafting_weights
                        do
                            for history_item in $history_items
                            do
                                for history_window in $history_windows
                                do
                                    for history_filter_top1_match in $history_filter_top1_matchs
                                    do
                                        for history_w_grid_measure in $history_w_grid_measures
                                        do
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    multi_turn_task=$multi_turn_task \
                                                    tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure exp_title=$exp_title ;
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MTP-history-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="TabedMTP"
eval_drafting="EvalTabedMTP" # align with draftings
tabed_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for temperature_drafting_weight in $temperature_drafting_weights
do
    for drf in $drfs
    do
        for tgt in $tgts
        do
            for dataset in $WholeData
            do
                for drafting in $draftings
                do
                    for tabed_rule in $tabed_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                for history_item in $history_items
                                do
                                    for history_window in $history_windows
                                    do
                                        for history_filter_top1_match in $history_filter_top1_matchs
                                        do
                                            for history_w_grid_measure in $history_w_grid_measures
                                            do
                                                for multi_turn_task in $multi_turn_tasks
                                                do
                                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                        drf=$drf tgt=$tgt \
                                                        $drafting \
                                                        tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                                        temperature_drafting_weight=$temperature_drafting_weight \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MTCP-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm" # align with drfs, tgts
# DRAFTING
draftings="TabedMTCP"
eval_drafting="EvalTabedMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
tabed_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for caption_type in $caption_types
                do
                    for captioning_model in $captioning_models
                    do
                        for tabed_rule in $tabed_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    for multi_turn_task in $multi_turn_tasks
                                    do
                                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                                            drf=$drf tgt=$tgt \
                                            $drafting \
                                            tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                            caption_type=$caption_type captioning_model=$captioning_model \
                                            target_dim_image_pooling=$target_dim_image_pooling \
                                            multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



device_num=0
exp_title="fp16-lvlm-7b-IbED-MTCP-history-multi-turn-999999"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm" # align with drfs, tgts
# DRAFTING
draftings="TabedMTCP"
eval_drafting="EvalTabedMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
tabed_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for caption_type in $caption_types
                do
                    for captioning_model in $captioning_models
                    do
                        for tabed_rule in $tabed_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    for history_item in $history_items
                                    do
                                        for history_window in $history_windows
                                        do
                                            for history_filter_top1_match in $history_filter_top1_matchs
                                            do
                                                for history_w_grid_measure in $history_w_grid_measures
                                                do
                                                    for multi_turn_task in $multi_turn_tasks
                                                    do
                                                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                            drf=$drf tgt=$tgt \
                                                            $drafting \
                                                            tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                                            caption_type=$caption_type captioning_model=$captioning_model \
                                                            target_dim_image_pooling=$target_dim_image_pooling \
                                                            history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                            history_w_grid_measure=$history_w_grid_measure \
                                                            multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
251121
3turn - single image
"""

WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
# WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
# eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=0
exp_title="fp16-lvlm-7b-single-M-rebuttal-turn3-251121"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
# DRAFTING
draftings="MultimodalDraft"
eval_drafting="multimodal"
# MULTITURN
multi_turn_tasks="qaorg-qaorg-eval"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for multi_turn_task in $multi_turn_tasks
                do
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-single-T-rebuttal-turn3-251121"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
draftings="TextOnlyDraft"
eval_drafting="text-only"
# MULTITURN
multi_turn_tasks="qaorg-qaorg-eval"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for multi_turn_task in $multi_turn_tasks
                do
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MT-history-rebuttal-turn3-251121"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="TabedMT"
eval_drafting="EvalTabedMT" # align with draftings
tabed_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg-eval"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for temperature_drafting_weight in $temperature_drafting_weights
                        do
                            for history_item in $history_items
                            do
                                for history_window in $history_windows
                                do
                                    for history_filter_top1_match in $history_filter_top1_matchs
                                    do
                                        for history_w_grid_measure in $history_w_grid_measures
                                        do
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
251121
3turn - multi image
"""

WholeData_ACL25="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=1
exp_title="fp16-lvlm-7b-single-M-rebuttal-turn3-251121"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
# DRAFTING
draftings="MultimodalDraft"
eval_drafting="multimodal"
# MULTITURN
multi_turn_tasks="qaorg-story-eval"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for multi_turn_task in $multi_turn_tasks
                do
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm-7b-single-T-rebuttal-turn3-251121"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
draftings="TextOnlyDraft"
eval_drafting="text-only"
# MULTITURN
multi_turn_tasks="qaorg-story-eval"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets
for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for multi_turn_task in $multi_turn_tasks
                do
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm-7b-IbED-MT-history-rebuttal-turn3-251121"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="TabedMT"
eval_drafting="EvalTabedMT" # align with draftings
tabed_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story-eval"
# DATASET
WholeData=$WholeData
eval_datasets=$eval_datasets

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for temperature_drafting_weight in $temperature_drafting_weights
                        do
                            for history_item in $history_items
                            do
                                for history_window in $history_windows
                                do
                                    for history_filter_top1_match in $history_filter_top1_matchs
                                    do
                                        for history_w_grid_measure in $history_w_grid_measures
                                        do
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"