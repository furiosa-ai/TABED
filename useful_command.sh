"""
250216: Second turn
MTC* grid	WIP 0218 A100:1
MTC* kld	WIP 0218 A100:2
MTP* grid	WIP 0218 A100:3
MTP* kld	WIP 0218 A6000:1
"""


"""MTC* grid"""

# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

# single image
device_num=1
exp_title="fp16-lvlm-7b-IbED-MTC-history-w-grid-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# double image
device_num=1
exp_title="fp16-lvlm-7b-IbED-MTC-history-w-grid-double-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""MTC* kld"""
# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

# single image
device_num=2
exp_title="fp16-lvlm-7b-IbED-MTC-history-kld-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# double image
device_num=2
exp_title="fp16-lvlm-7b-IbED-MTC-history-kld-double-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""MTP* w-grid"""

# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG


device_num=3
exp_title="fp16-lvlm-7b-IbED-MTP-history-w-grid-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld" # shit!!
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        temperature_drafting_weight=$temperature_drafting_weight \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        multi_turn_task=$multi_turn_task \
                                                        exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=3
exp_title="fp16-lvlm-7b-IbED-MTP-history-w-grid-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        temperature_drafting_weight=$temperature_drafting_weight \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        multi_turn_task=$multi_turn_task \
                                                        exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



"""MTP* kld"""
# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG


device_num=1
# exp_title="fp16-lvlm-7b-IbED-MTP-history-w-grid-single-image-second-turn-250218" => 돌렸으나 이름 잘못설정
exp_title="fp16-lvlm-7b-IbED-MTP-history-kld-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    multi_turn_task=$multi_turn_task \
                                                    exp_title=$exp_title ;
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
# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG


device_num=1
exp_title="fp16-lvlm-7b-IbED-MTP-history-w-grid-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
# exp_title="fp16-lvlm-7b-IbED-MTP-history-w-grid-single-image-second-turn-250218" => 돌렸으나 이름 잘못설정
exp_title="fp16-lvlm-7b-IbED-MTP-history-kld-double-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    multi_turn_task=$multi_turn_task \
                                                    exp_title=$exp_title ;
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
# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG
device_num=1
# exp_title="fp16-lvlm-7b-IbED-MTP-history-w-grid-single-image-second-turn-250218"
exp_title="fp16-lvlm-7b-IbED-MTP-history-kld-double-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""MTC* grid 13b"""

# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

# single image
device_num=1
exp_title="fp16-lvlm-13b-IbED-MTC-history-w-grid-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# double image
device_num=1
exp_title="fp16-lvlm-13b-IbED-MTC-history-w-grid-double-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""MTC* kld 13b"""
# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

# single image
device_num=2
exp_title="fp16-lvlm-13b-IbED-MTC-history-kld-single-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# double image
device_num=2
exp_title="fp16-lvlm-13b-IbED-MTC-history-kld-double-image-second-turn-250218"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""MTP* kld"""
# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG


device_num=1
exp_title="fp16-lvlm-13b-IbED-MTP-history-kld-single-image-second-turn-250219"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    multi_turn_task=$multi_turn_task \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm-13b-IbED-MTP-history-kld-double-image-second-turn-250219"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    multi_turn_task=$multi_turn_task \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
250304
MTC* 13b grid/kld
"""

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=2
exp_title="fp16-lvlm-13b-IbED-MTC-history-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
# multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
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
                for cascade_rule in $cascade_rules
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
                                            # for multi_turn_task in $multi_turn_tasks
                                            # do
                                                # multi_turn_task=$multi_turn_task \
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
                                            # done
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"

"""
250304
MTP* 13b grid/kld
"""

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm-13b-IbED-MTP-history-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
# multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
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
                    for cascade_rule in $cascade_rules
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
                                            # for multi_turn_task in $multi_turn_tasks
                                            # do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    exp_title=$exp_title ;
                                            # done
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
250304
MTP* grid second turn
"""
# Single image in ACL25 (4)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# Double image in ACL25 (3)
WholeData_ACL25_DOUBLE_IMG="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG


device_num=1
exp_title="fp16-lvlm-13b-IbED-MTP-history-w-grid-single-image-second-turn-250219"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        temperature_drafting_weight=$temperature_drafting_weight \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        multi_turn_task=$multi_turn_task \
                                                        exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm-13b-IbED-MTP-history-kld-double-image-second-turn-250219"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-story qaorg-nq qaorg-gsm8k"
# DATASET
WholeData=$WholeData_ACL25_DOUBLE_IMG
eval_datasets=$eval_datasets_ACL25_DOUBLE_IMG
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
                    for cascade_rule in $cascade_rules
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
                                            for multi_turn_task in $multi_turn_tasks
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    multi_turn_task=$multi_turn_task \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_multi_turn_task=$eval_multi_turn_task \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



# 250304 NextVicuna7b M

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm-NextVicuna7b-single-M-250304"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b"
draftings="MultimodalDraft"
eval_drafting="multimodal"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250304 NextVicuna7b T

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm-NextVicuna7b-single-T-250304"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b"
draftings="TextOnlyDraft"
eval_drafting="text-only"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250304 NextVicuna7b MT

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MT-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250304 NextVicuna7b MT*

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MT-history-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlmNextVicuna7b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



"""
250304
C, P, MTC, MTP, MTC*, MTP*
"""

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

# 250304 C
device_num=2
exp_title="fp16-lvlm-NextVicuna7b-single-C-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CaptionDraft"
eval_drafting="caption" # align with draftings
caption_types="<CAPTION>" # caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
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
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            caption_type=$caption_type captioning_model=$captioning_model \
                            exp_title=$exp_title ;
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



# 250304 P
device_num=2
exp_title="fp16-lvlm-NextVicuna7b-single-P-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="InferencePoolDraft"
eval_drafting="image-pool"
target_dim_image_poolings="144"
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
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        target_dim_image_pooling=$target_dim_image_pooling \
                        exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250304 MTC

device_num=2
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTC-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=False
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            confidence_type=$confidence_type \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250304 MTP

device_num=2
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTP-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=False
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
                    for cascade_rule in $cascade_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                    drf=$drf tgt=$tgt \
                                    $drafting \
                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                    target_dim_image_pooling=$target_dim_image_pooling \
                                    exp_title=$exp_title ;
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250304 MTC* 
device_num=2
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTC-history-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250304 MTP*

device_num=2
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTP-history-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                    for cascade_rule in $cascade_rules
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
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
250304
LlavaNextVicuna7b
MTCP, MTCP*
"""

WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25
# WholeData=$WholeData
# eval_datasets=$eval_datasets

# MTCP
device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTCP-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                        for cascade_rule in $cascade_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                        drf=$drf tgt=$tgt \
                                        $drafting \
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                        caption_type=$caption_type captioning_model=$captioning_model \
                                        target_dim_image_pooling=$target_dim_image_pooling \
                                        exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# MTCP*
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTCP-history-250304"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                        for cascade_rule in $cascade_rules
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
                                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                        drf=$drf tgt=$tgt \
                                                        $drafting \
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        caption_type=$caption_type captioning_model=$captioning_model \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        exp_title=$exp_title;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
250305
lvlmOV - LlavaNextVicuna7b
M, T, MT, MT*
C, P, MTC, MTP, MTC*, MTP*
MTCP, MTCP*
"""

# 250305 NextVicuna7b M

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-single-M-250305"
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b"
draftings="MultimodalDraft"
eval_drafting="multimodal"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 NextVicuna7b T

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-single-T-250305"
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b"
draftings="TextOnlyDraft"
eval_drafting="text-only"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 NextVicuna7b MT

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MT-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 NextVicuna7b MT*

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MT-history-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlmOvNextVicuna7b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"





# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

# 250305 C
device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-single-C-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CaptionDraft"
eval_drafting="caption" # align with draftings
caption_types="<CAPTION>" # caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
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
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            caption_type=$caption_type captioning_model=$captioning_model \
                            exp_title=$exp_title ;
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



# 250305 P
device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-single-P-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="InferencePoolDraft"
eval_drafting="image-pool"
target_dim_image_poolings="144"
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
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        target_dim_image_pooling=$target_dim_image_pooling \
                        exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTC

device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MTC-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=False
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            confidence_type=$confidence_type \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTP

device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MTP-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=False
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
                    for cascade_rule in $cascade_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                    drf=$drf tgt=$tgt \
                                    $drafting \
                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                    target_dim_image_pooling=$target_dim_image_pooling \
                                    exp_title=$exp_title ;
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTC* 
device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MTC-history-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTP*

device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MTP-history-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                    for cascade_rule in $cascade_rules
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
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25
# WholeData=$WholeData
# eval_datasets=$eval_datasets

# MTCP
device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MTCP-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                        for cascade_rule in $cascade_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                        drf=$drf tgt=$tgt \
                                        $drafting \
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                        caption_type=$caption_type captioning_model=$captioning_model \
                                        target_dim_image_pooling=$target_dim_image_pooling \
                                        exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# MTCP*
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=3
exp_title="fp16-lvlm68m-ov-NextVicuna7b-IbED-MTCP-history-250305"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                        for cascade_rule in $cascade_rules
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
                                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                        drf=$drf tgt=$tgt \
                                                        $drafting \
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        caption_type=$caption_type captioning_model=$captioning_model \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        exp_title=$exp_title;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"

"""
250305
lvlm160m - LlavaNextVicuna7b
M, T, MT, MT*
C, P, MTC, MTP, MTC*, MTP*
MTCP, MTCP*
"""

# 250305 NextVicuna7b M

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-single-M-250305"
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b"
draftings="MultimodalDraft"
eval_drafting="multimodal"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 NextVicuna7b T

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-single-T-250305"
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b"
draftings="TextOnlyDraft"
eval_drafting="text-only"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 NextVicuna7b MT

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MT-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 NextVicuna7b MT*

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MT-history-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm160mBfNextVicuna7b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"





# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

# 250305 C
device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-single-C-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CaptionDraft"
eval_drafting="caption" # align with draftings
caption_types="<CAPTION>" # caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
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
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                            $drafting \
                            caption_type=$caption_type captioning_model=$captioning_model \
                            exp_title=$exp_title ;
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
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



# 250305 P
device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-single-P-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="InferencePoolDraft"
eval_drafting="image-pool"
target_dim_image_poolings="144"
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
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                        $drafting \
                        target_dim_image_pooling=$target_dim_image_pooling \
                        exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTC

device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTC-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=False
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            confidence_type=$confidence_type \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTP

device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTP-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=False
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
                    for cascade_rule in $cascade_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                    drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                    $drafting \
                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                    target_dim_image_pooling=$target_dim_image_pooling \
                                    exp_title=$exp_title ;
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTC* 
device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTC-history-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250305 MTP*

device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTP-history-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                    for cascade_rule in $cascade_rules
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
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    target_dim_image_pooling=$target_dim_image_pooling \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25
# WholeData=$WholeData
# eval_datasets=$eval_datasets

# MTCP
device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTCP-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                        for cascade_rule in $cascade_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                        drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                        $drafting \
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                        caption_type=$caption_type captioning_model=$captioning_model \
                                        target_dim_image_pooling=$target_dim_image_pooling \
                                        exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# MTCP*
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=2
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTCP-history-250305"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                        for cascade_rule in $cascade_rules
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
                                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                        drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                        $drafting \
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        caption_type=$caption_type captioning_model=$captioning_model \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        exp_title=$exp_title;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
250305
MTC* 
"""


device_num=1
exp_title="fp16-lvlm-7b-IbED-MTC-history-250305"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"




"""
250305
Visualize ..
"""
# w-grid num-accepted (multiple)
python3 mllmsd/utils/oracle_real_calculate.py --objective=w-grid --draftings=multimodal,text-only --history_w_grid_measure=num-accepted --history_w_grid_num_accepted_order=none

# w-grid num-accepted-distance
python3 mllmsd/utils/oracle_real_calculate.py --objective=w-grid --draftings=multimodal,text-only --history_w_grid_measure=num-accepted-kld


"""
250306
MT*
"""





"""
250306 13b 채우기
ACL25: MT*, MTP, MTCP*
ACL25-일부: M, T, MT, C, P, MTC, MTCP
"""
# 250306  M

# ACL25 (9)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

device_num=1

exp_title="fp16-lvlm68m-13b-single-M-250306"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b"
draftings="MultimodalDraft"
eval_drafting="multimodal"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250306  T

# ACL25 (9)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

device_num=1
exp_title="fp16-lvlm68m-13b-single-T-250306"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b"
draftings="TextOnlyDraft"
eval_drafting="text-only"
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
                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                    drf=$drf tgt=$tgt \
                    $drafting \
                    exp_title=$exp_title ;
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250306  MT

# ACL25 (9)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

device_num=1
exp_title="fp16-lvlm68m-13b-IbED-MT-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"





# ACL25 (9)
WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG

# 250306 C
device_num=1
exp_title="fp16-lvlm68m-13b-single-C-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CaptionDraft"
eval_drafting="caption" # align with draftings
caption_types="<CAPTION>" # caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
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
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            caption_type=$caption_type captioning_model=$captioning_model \
                            exp_title=$exp_title ;
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



# 250306 P
device_num=1
exp_title="fp16-lvlm68m-13b-single-P-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="InferencePoolDraft"
eval_drafting="image-pool"
target_dim_image_poolings="144"
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
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt \
                        $drafting \
                        target_dim_image_pooling=$target_dim_image_pooling \
                        exp_title=$exp_title ;
                done
            done
        done
    done
done
eval_drafting=$(echo $eval_drafting | sed 's/ /,/g')
eval_datasets=$(echo $eval_datasets | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250306 MTC

device_num=1
exp_title="fp16-lvlm68m-13b-IbED-MTC-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=False
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                            confidence_type=$confidence_type \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"




WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG
# WholeData=$WholeData
# eval_datasets=$eval_datasets

# MTCP
device_num=1
exp_title="fp16-lvlm68m-13b-IbED-MTCP-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
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
                        for cascade_rule in $cascade_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                        drf=$drf tgt=$tgt \
                                        $drafting \
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                        caption_type=$caption_type captioning_model=$captioning_model \
                                        target_dim_image_pooling=$target_dim_image_pooling \
                                        exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"




# 250306  MT*

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=1
exp_title="fp16-lvlm68m-13b-IbED-MT-history-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                for cascade_rule in $cascade_rules
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt \
                                                $drafting \
                                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


# 250306 MTP

device_num=1
exp_title="fp16-lvlm68m-13b-IbED-MTP-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=False
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
                    for cascade_rule in $cascade_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                    drf=$drf tgt=$tgt \
                                    $drafting \
                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                    target_dim_image_pooling=$target_dim_image_pooling \
                                    exp_title=$exp_title ;
                            done
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"

# MTCP*
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=1
exp_title="fp16-lvlm68m-13b-IbED-MTCP-history-250306"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                        for cascade_rule in $cascade_rules
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
                                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                        drf=$drf tgt=$tgt \
                                                        $drafting \
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        caption_type=$caption_type captioning_model=$captioning_model \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        exp_title=$exp_title;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


"""
250310 
68m - NextVicuna7b
Multiturn - ALL
"""

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=3
exp_title="fp16-lvlm-NextVicuna7b-single-M-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-single-T-multi-turn-250310"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MT-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MT-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-single-C-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
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


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-single-P-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTC-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                confidence_type=$confidence_type \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTP-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
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
                    for cascade_rule in $cascade_rules
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
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"

device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTC-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTP-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
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
                    for cascade_rule in $cascade_rules
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
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTCP-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
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
                        for cascade_rule in $cascade_rules
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
                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



device_num=3
exp_title="fp16-lvlm-NextVicuna7b-IbED-MTCP-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
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
                        for cascade_rule in $cascade_rules
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
                                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
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
250310 
ov68m - NextVicuna7b
Multiturn - ALL
"""

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-single-M-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-single-T-multi-turn-250310"
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MT-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MT-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-single-C-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
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


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-single-P-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MTC-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                confidence_type=$confidence_type \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MTP-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
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
                    for cascade_rule in $cascade_rules
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
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"

device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MTC-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MTP-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
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
                    for cascade_rule in $cascade_rules
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
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MTCP-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
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
                        for cascade_rule in $cascade_rules
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
                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



device_num=2
exp_title="fp16-lvlm-ov-NextVicuna7b-IbED-MTCP-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm68m-ov"
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlmOvNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
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
                        for cascade_rule in $cascade_rules
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
                                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
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
250310 
160m - NextVicuna7b
Multiturn - ALL
"""

# ACL25 (9)
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-single-M-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b"
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
                        drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-single-T-multi-turn-250310"
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b"
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
                        drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MT-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MT-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
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
                                                    drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-single-C-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
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
                                drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
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


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-single-P-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
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
                            drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTC-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                confidence_type=$confidence_type \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTP-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
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
                    for cascade_rule in $cascade_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                for multi_turn_task in $multi_turn_tasks
                                do
                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                        drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                        $drafting \
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"

device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTC-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
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
                for cascade_rule in $cascade_rules
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
                                                    drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                    $drafting \
                                                    multi_turn_task=$multi_turn_task \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTP-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
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
                    for cascade_rule in $cascade_rules
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
                                                        drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                        $drafting \
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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


device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTCP-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
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
                        for cascade_rule in $cascade_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    for multi_turn_task in $multi_turn_tasks
                                    do
                                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                                            drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                            $drafting \
                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



device_num=1
exp_title="fp16-lvlm160m-NextVicuna7b-IbED-MTCP-history-multi-turn-250310"
# MODEL
drfs="mjbooo/lvlm160m-bf16"; drf_dtype="bf16"; tgt_dtype="bf16";
tgts="llava-hf/llava-v1.6-vicuna-7b-hf"
eval_model="EvalLvlm160mBfNextVicuna7b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
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
                        for cascade_rule in $cascade_rules
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
                                                            drf=$drf tgt=$tgt drf_dtype=$drf_dtype tgt_dtype=$tgt_dtype \
                                                            $drafting \
                                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
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
250329
13b multiturn 
"""

WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=0
exp_title="fp16-lvlm-7b-single-M-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b"
# DRAFTING
draftings="MultimodalDraft"
eval_drafting="multimodal"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-single-T-multi-turn-250329"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b"
draftings="TextOnlyDraft"
eval_drafting="text-only"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MT-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=0
exp_title="fp16-lvlm-7b-IbED-MT-history-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld tvd" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
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





WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=1
exp_title="fp16-lvlm-7b-single-C-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CaptionDraft"
eval_drafting="caption" # align with draftings
caption_types="<CAPTION>" # caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
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


device_num=1
exp_title="fp16-lvlm-7b-single-P-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="InferencePoolDraft"
eval_drafting="image-pool"
target_dim_image_poolings="144"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=1
exp_title="fp16-lvlm-7b-IbED-MTC-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTC"
eval_drafting="EvalCascadeMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=False
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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
                for cascade_rule in $cascade_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        for multi_turn_task in $multi_turn_tasks
                        do
                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                drf=$drf tgt=$tgt \
                                $drafting \
                                cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                confidence_type=$confidence_type \
                                multi_turn_task=$multi_turn_task exp_title=$exp_title ;
                        done
                    done
                done
            done
        done
    done
done
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_captioning_model=$eval_captioning_model \
    eval_caption_type=$eval_caption_type \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=2
exp_title="fp16-lvlm-7b-IbED-MTP-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTP"
eval_drafting="EvalCascadeMTP" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
target_dim_image_poolings="144"
# HISTORY
history_dependent=False
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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
                    for cascade_rule in $cascade_rules
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
                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


device_num=2
exp_title="fp16-lvlm-7b-IbED-MTCP-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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
                        for cascade_rule in $cascade_rules
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
                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=3
exp_title="fp16-lvlm-7b-IbED-MTCP-history-multi-turn-250329"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# MULTITURN
multi_turn_tasks="qaorg-qaorg qaorg-story qaorg-nq qaorg-gsm8k"
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
                        for cascade_rule in $cascade_rules
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
                                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
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
250920
MTCP* random
"""

WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=0,1
exp_title="fp16-lvlm-7b-IbED-MTCP-history-random-250920"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="random" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                        for cascade_rule in $cascade_rules
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
                                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                        drf=$drf tgt=$tgt \
                                                        $drafting \
                                                        cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                        caption_type=$caption_type captioning_model=$captioning_model \
                                                        target_dim_image_pooling=$target_dim_image_pooling \
                                                        history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                        history_w_grid_measure=$history_w_grid_measure \
                                                        exp_title=$exp_title;
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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
python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"




fp16-lvlm68m-7b-IbED-MT-history-random-250918
fp16-lvlm-7b-IbED-MTCP-history-random-250920


"""
251004
random seconturn - MT*, MTCP*
"""
WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

# A100
device_num=0,1,2,3,4,5,6,7
exp_title="fp16-lvlm-7b-IbED-MT-history-random-multi-turn-251004"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="random" # history_items="kld tvd w-grid random"; 
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
                for cascade_rule in $cascade_rules
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
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_multi_turn_task=$(echo $multi_turn_tasks | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/mnt-legacy/home-mjlee/data/MSD/csv"
git_dir="/mnt-legacy/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"




# H100

WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=0,1,2,3
exp_title="fp16-lvlm-7b-IbED-MTCP-history-random-multi-turn-251004"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm" # align with drfs, tgts
# DRAFTING
draftings="CascadeMTCP"
eval_drafting="EvalCascadeMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
cascade_rules="mm-weight"
mm_weight_policys="1" # mm_weight_policys="1 2 3 4"
# HISTORY
history_dependent=True
history_windows="0" # history_windows="0 1 4 16"
history_items="random" # history_items="kld tvd w-grid random"; 
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
                        for cascade_rule in $cascade_rules
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
                                                            cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
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
eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/mnt-legacy/home-mjlee/data/MSD/csv"
git_dir="/mnt-legacy/home-mjlee/data/MSD/MLLMSD_Results"

cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


/mnt-legacy/home-mjlee/workspace/MLLMSD

conda create -n msd2 python=3.10
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 ;
pip3 install transformers==4.44.0 absl-py==2.1.0 wandb sacred==0.8.5 datasets==2.20.0 accelerate==0.28.0 einops==0.7.0 scipy joblib==1.4.0  sentencepiece==0.2.0 matplotlib timm ;
pip install flash-attn==2.6.2 --no-build-isolation ;

huggingface-cli login


fp16-lvlm-7b-IbED-MT-history-random-multi-turn-251004
fp16-lvlm-7b-IbED-MTCP-history-random-multi-turn-251004




"""
251119
adaboost
"""

WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"

device_num=0,1
exp_title="fp16-lvlm-7b-IbED-MT-history-251119"
# MODEL
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# DRAFTING
draftings="CascadeMT"
eval_drafting="EvalCascadeMT" # align with draftings
cascade_rules="mm-weight"
mm_weight_policys="1"
# HISTORY
history_dependent=True
history_windows="0 1 4 16" # history_windows="0 1 4 16"
history_items="adaboost" # history_items="kld tvd w-grid"; 
history_adaboost_constant_weights="1 0.5 0.25 0.125"
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# DATASET
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

for drf in $drfs
do
    for tgt in $tgts
    do
        for dataset in $WholeData
        do
            for drafting in $draftings
            do
                for cascade_rule in $cascade_rules
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
                                            for history_adaboost_constant_weight in $history_adaboost_constant_weights
                                            do
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
                                                    confidence_type=$confidence_type \
                                                    temperature_drafting_weight=$temperature_drafting_weight \
                                                    history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                    history_w_grid_measure=$history_w_grid_measure \
                                                    history_adaboost_constant_weight=$history_adaboost_constant_weight \
                                                    exp_title=$exp_title ;
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

eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
eval_history_item=$(echo $history_items | sed 's/ /,/g')
eval_history_window=$(echo $history_windows | sed 's/ /,/g')
eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')
eval_history_adaboost_constant_weight=$(echo $history_adaboost_constant_weights | sed 's/ /,/g')

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_cascade_rule=$eval_cascade_rule \
    eval_mm_weight_policy=$eval_mm_weight_policy \
    eval_history_dependent=$eval_history_dependent \
    eval_history_item=$eval_history_item \
    eval_history_window=$eval_history_window \
    eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
    eval_history_w_grid_measure=$eval_history_w_grid_measure \
    eval_history_filter_top1_match=$eval_history_filter_top1_match \
    eval_history_adaboost_constant_weight=$eval_history_adaboost_constant_weight ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"




# qaorg-qaorg
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"

# qaorg-story
eval_datasets_ACL25_DOUBLE_IMG="IEdit,MagicBrush,Spot-the-Diff"



"""
251121
3turn 
"""

WholeData_ACL25="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25

device_num=0
exp_title="fp16-lvlm-7b-single-T-three-turn-data-251121"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
draftings="TextOnlyDraft"
eval_drafting="text-only"
# MULTITURN
multi_turn_tasks="qaorg-qaorg"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"


##########################################
WholeData_ACL25="IEditData MagicBrushData SpotTheDiffData"
eval_datasets_ACL25="IEdit,MagicBrush,Spot-the-Diff"
WholeData=$WholeData_ACL25
eval_datasets=$eval_datasets_ACL25


device_num=1
exp_title="fp16-lvlm-7b-single-T-three-turn-data-251121"
drfs="mjbooo/lvlm68m"
tgts="llava-hf/llava-1.5-7b-hf"
eval_model="EvalLvlm"
draftings="TextOnlyDraft"
eval_drafting="text-only"
# MULTITURN
multi_turn_tasks="qaorg-story"
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

python3 mllmsd/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    eval_drafting=$eval_drafting \
    eval_datasets=$eval_datasets \
    eval_multi_turn_task=$eval_multi_turn_task ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



fp16-lvlm-7b-IbED-MT-history-rebuttal-turn3-251121
fp16-lvlm-7b-single-M-rebuttal-turn3-251121
fp16-lvlm-7b-single-T-rebuttal-turn3-251121






# device_num=1
# exp_title="fp16-lvlm-7b-IbED-MTC-history-250305"
# # MODEL
# drfs="mjbooo/lvlm68m"
# tgts="llava-hf/llava-1.5-7b-hf" # tgts="llava-hf/llava-1.5-13b-hf"
# eval_model="EvalLvlm" # eval_model="EvalLvlm13b" # align with drfs, tgts
# # DRAFTING
# draftings="CascadeMTC"
# eval_drafting="EvalCascadeMTC" # align with draftings
# captioning_models="microsoft/Florence-2-large-ft"
# caption_types="<CAPTION>"
# cascade_rules="mm-weight"
# mm_weight_policys="1"
# # HISTORY
# history_dependent=True
# history_windows="0" # history_windows="0 1 4 16"
# history_items="kld w-grid" # history_items="kld tvd w-grid"; 
# history_w_grid_measures="num-accepted-kld"
# temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
# history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
# # DATASET
# WholeData=$WholeData
# eval_datasets=$eval_datasets

# for drf in $drfs
# do
#     for tgt in $tgts
#     do
#         for dataset in $WholeData
#         do
#             for drafting in $draftings
#             do
#                 for cascade_rule in $cascade_rules
#                 do
#                     for mm_weight_policy in $mm_weight_policys
#                     do
#                         for temperature_drafting_weight in $temperature_drafting_weights
#                         do
#                             for history_item in $history_items
#                             do
#                                 for history_window in $history_windows
#                                 do
#                                     for history_filter_top1_match in $history_filter_top1_matchs
#                                     do
#                                         for history_w_grid_measure in $history_w_grid_measures
#                                         do
#                                             OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
#                                                 python3 main.py with SpecDecoding HalfPrecision $dataset \
#                                                 drf=$drf tgt=$tgt \
#                                                 $drafting \
#                                                 cascade_rule=$cascade_rule mm_weight_policy=$mm_weight_policy \
#                                                 confidence_type=$confidence_type \
#                                                 temperature_drafting_weight=$temperature_drafting_weight \
#                                                 history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
#                                                 history_w_grid_measure=$history_w_grid_measure \
#                                                 exp_title=$exp_title ;
#                                         done
#                                     done
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
# eval_cascade_rule=$(echo $cascade_rules | sed 's/ /,/g')
# eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
# eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
# eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
# eval_history_dependent=$(echo $history_dependent | sed 's/ /,/g')
# eval_history_item=$(echo $history_items | sed 's/ /,/g')
# eval_history_window=$(echo $history_windows | sed 's/ /,/g')
# eval_temperature_drafting_weight=$(echo $temperature_drafting_weights | sed 's/ /,/g')
# eval_history_filter_top1_match=$(echo $history_filter_top1_matchs | sed 's/ /,/g')
# eval_history_w_grid_measure=$(echo $history_w_grid_measures | sed 's/ /,/g')

# python3 mllmsd/utils/evaluation.py with EvaluationSD \
#     exp_title=$exp_title \
#     $eval_model \
#     $eval_drafting \
#     confidence_type=$confidence_type \
#     eval_datasets=$eval_datasets \
#     eval_cascade_rule=$eval_cascade_rule \
#     eval_captioning_model=$eval_captioning_model \
#     eval_caption_type=$eval_caption_type \
#     eval_mm_weight_policy=$eval_mm_weight_policy \
#     eval_history_dependent=$eval_history_dependent \
#     eval_history_item=$eval_history_item \
#     eval_history_window=$eval_history_window \
#     eval_temperature_drafting_weight=$eval_temperature_drafting_weight \
#     eval_history_w_grid_measure=$eval_history_w_grid_measure \
#     eval_history_filter_top1_match=$eval_history_filter_top1_match ;

# csv_dir="/pvc/home-mjlee/data/MSD/csv"
# git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
# cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"





# Source environment variables (optional - if you want to use configurable paths)
# source setup_env.sh

WholeData_ACL25_SINGLE_IMG="LlavaBenchInTheWildData DocVQAData PopeData MMVetData"
eval_datasets_ACL25_SINGLE_IMG="llava-bench-in-the-wild,docvqa_val,POPE,MMVet"
WholeData=$WholeData_ACL25_SINGLE_IMG
eval_datasets=$eval_datasets_ACL25_SINGLE_IMG


device_num=1
exp_title="fp16-lvlm-7b-IbED-MTC-history-250305"
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
history_items="kld w-grid" # history_items="kld tvd w-grid"; 
history_w_grid_measures="num-accepted-kld"
temperature_drafting_weights="1" # temperature_drafting_weights="1 0.75 0.5 0.25"
history_filter_top1_matchs="False" # history_filter_top1_matchs="True False"
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
                                            OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                drf=$drf tgt=$tgt \
                                                $drafting \
                                                tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                                                confidence_type=$confidence_type \
                                                temperature_drafting_weight=$temperature_drafting_weight \
                                                history_dependent=$history_dependent history_item=$history_item history_window=$history_window history_filter_top1_match=$history_filter_top1_match \
                                                history_w_grid_measure=$history_w_grid_measure \
                                                exp_title=$exp_title ;
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
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

# Use environment variables for paths (set via setup_env.sh or defaults)
csv_dir="${TABED_RESULTS_DIR:-$HOME/data/tabed/results}/csv"
git_dir="${TABED_RESULTS_DIR:-$HOME/data/tabed/results}"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"



- no trainer
- path 

/pvc/home-mjlee/data/MSD

checkpoint
npy
MLLMSD_Results
datasets