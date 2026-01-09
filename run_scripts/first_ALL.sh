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
python3 tabed/utils/evaluation.py with EvaluationSD \
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
python3 tabed/utils/evaluation.py with EvaluationSD \
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
draftings="TabedMT"
eval_drafting="EvalTabedMT" # align with draftings
tabed_rules="mm-weight"
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
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                            exp_title=$exp_title ;
                    done
                done
            done
        done
    done
done
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
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
python3 tabed/utils/evaluation.py with EvaluationSD \
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
python3 tabed/utils/evaluation.py with EvaluationSD \
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
draftings="TabedMTC"
eval_drafting="EvalTabedMTC" # align with draftings
captioning_models="microsoft/Florence-2-large-ft"
caption_types="<CAPTION>"
tabed_rules="mm-weight"
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
                for tabed_rule in $tabed_rules
                do
                    for mm_weight_policy in $mm_weight_policys
                    do
                        OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                            python3 main.py with SpecDecoding HalfPrecision $dataset \
                            drf=$drf tgt=$tgt \
                            $drafting \
                            tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
                            confidence_type=$confidence_type \
                            exp_title=$exp_title ;
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

python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    confidence_type=$confidence_type \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
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
draftings="TabedMTP"
eval_drafting="EvalTabedMTP" # align with draftings
tabed_rules="mm-weight"
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
                    for tabed_rule in $tabed_rules
                    do
                        for mm_weight_policy in $mm_weight_policys
                        do
                            for target_dim_image_pooling in $target_dim_image_poolings
                            do
                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                    drf=$drf tgt=$tgt \
                                    $drafting \
                                    tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
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
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_tabed_rule=$eval_tabed_rule \
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
draftings="TabedMTP"
eval_drafting="EvalTabedMTP" # align with draftings
tabed_rules="mm-weight"
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
                                                OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                    python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                    drf=$drf tgt=$tgt \
                                                    $drafting \
                                                    tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
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
draftings="TabedMTCP"
eval_drafting="EvalTabedMTCP" # align with draftings
caption_types="<CAPTION>"
# caption_types="<CAPTION> <MORE_DETAILED_CAPTION>"
captioning_models="microsoft/Florence-2-large-ft"
target_dim_image_poolings="144"
tabed_rules="mm-weight"
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
                        for tabed_rule in $tabed_rules
                        do
                            for mm_weight_policy in $mm_weight_policys
                            do
                                for target_dim_image_pooling in $target_dim_image_poolings
                                do
                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                        drf=$drf tgt=$tgt \
                                        $drafting \
                                        tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
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
eval_tabed_rule=$(echo $tabed_rules | sed 's/ /,/g')
eval_mm_weight_policy=$(echo $mm_weight_policys | sed 's/ /,/g')
eval_caption_type=$(echo $caption_types | sed 's/ /,/g')
eval_captioning_model=$(echo $captioning_models | sed 's/ /,/g')
eval_target_dim_image_pooling=$(echo $target_dim_image_poolings | sed 's/ /,/g')
python3 tabed/utils/evaluation.py with EvaluationSD \
    exp_title=$exp_title \
    $eval_model \
    $eval_drafting \
    eval_datasets=$eval_datasets \
    eval_caption_type=$eval_caption_type \
    eval_captioning_model=$eval_captioning_model \
    eval_target_dim_image_pooling=$eval_target_dim_image_pooling \
    eval_tabed_rule=$eval_tabed_rule \
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
                                                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                                                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                                                        drf=$drf tgt=$tgt \
                                                        $drafting \
                                                        tabed_rule=$tabed_rule mm_weight_policy=$mm_weight_policy \
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
    eval_history_filter_top1_match=$eval_history_filter_top1_match ;

csv_dir="/pvc/home-mjlee/data/MSD/csv"
git_dir="/pvc/home-mjlee/data/MSD/MLLMSD_Results"
cp -r "$csv_dir/$exp_title" "$git_dir/$exp_title"