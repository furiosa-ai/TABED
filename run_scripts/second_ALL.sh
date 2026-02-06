#!/bin/bash
# TABED Evaluation Script - Multi-turn experiments

# =============================================================================
# Configuration
# =============================================================================
device_num=0
datasets=(LlavaBenchInTheWildData DocVQAData PopeData MMVetData IEditData MagicBrushData SpotTheDiffData PororoSVData VISTData)
eval_datasets="llava-bench-in-the-wild,docvqa_val,POPE,MMVet,IEdit,MagicBrush,Spot-the-Diff,PororoSV,VIST"

# Model configuration
drfs=(mjbooo/lvlm68m)
tgts=(llava-hf/llava-1.5-7b-hf)
eval_model="EvalLvlm"

# TABED configuration
tabed_rules=(mm-weight)
mm_weight_policys=(1)
captioning_models=(microsoft/Florence-2-large-ft)
caption_types=("<CAPTION>")
target_dim_image_poolings=(144)

# History configuration
history_windows=(0)
history_items=(w-grid kld tvd)
history_w_grid_measures=(num-accepted-kld)

# Multi-turn tasks
multi_turn_tasks=(qaorg-qaorg qaorg-captioning qaorg-summary qaorg-story qaorg-nq qaorg-gsm8k)

# =============================================================================
# Helper Functions
# =============================================================================
run_experiment() {
    local exp_title=$1
    local drafting=$2
    local extra_args="${@:3}"

    for drf in "${drfs[@]}"; do
        for tgt in "${tgts[@]}"; do
            for dataset in "${datasets[@]}"; do
                for multi_turn_task in "${multi_turn_tasks[@]}"; do
                    OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 CUDA_VISIBLE_DEVICES=$device_num \
                        python3 main.py with SpecDecoding HalfPrecision $dataset \
                        drf=$drf tgt=$tgt $drafting \
                        multi_turn_task=$multi_turn_task $extra_args exp_title=$exp_title
                done
            done
        done
    done
}

run_evaluation() {
    local exp_title=$1
    local eval_drafting=$2
    shift 2
    local extra_args="$@"

    python3 tabed/utils/evaluation.py with EvaluationSD \
        exp_title=$exp_title $eval_model $eval_drafting \
        eval_datasets=$eval_datasets \
        eval_multi_turn_task=$(IFS=,; echo "${multi_turn_tasks[*]}") \
        $extra_args
}

# =============================================================================
# Single Drafting Experiments (M, T, C, P)
# =============================================================================

# M: Multimodal
exp_title="llava15-68m-7b-single-M-second-turn"
run_experiment "$exp_title" "MultimodalDraft"
run_evaluation "$exp_title" "eval_drafting=multimodal"

# T: Text-only
exp_title="llava15-68m-7b-single-T-second-turn"
run_experiment "$exp_title" "TextOnlyDraft"
run_evaluation "$exp_title" "eval_drafting=text-only"

# C: Caption
exp_title="llava15-68m-7b-single-C-second-turn"
for caption_type in "${caption_types[@]}"; do
    for captioning_model in "${captioning_models[@]}"; do
        run_experiment "$exp_title" "CaptionDraft" \
            "caption_type=$caption_type" "captioning_model=$captioning_model"
    done
done
run_evaluation "$exp_title" "eval_drafting=caption" \
    "eval_caption_type=${caption_types[*]}" "eval_captioning_model=${captioning_models[*]}"

# P: Image Pool
exp_title="llava15-68m-7b-single-P-second-turn"
for target_dim in "${target_dim_image_poolings[@]}"; do
    run_experiment "$exp_title" "InferencePoolDraft" "target_dim_image_pooling=$target_dim"
done
run_evaluation "$exp_title" "eval_drafting=image-pool" \
    "eval_target_dim_image_pooling=${target_dim_image_poolings[*]}"

# =============================================================================
# TABED Ensemble Experiments (MT, MTC, MTP, MTCP)
# =============================================================================

# MT: Multimodal + Text-only
exp_title="llava15-68m-7b-IbED-MT-second-turn"
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        run_experiment "$exp_title" "TabedMT" \
            "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy"
    done
done
run_evaluation "$exp_title" "EvalTabedMT" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}"

# MT*: MT with history
exp_title="llava15-68m-7b-IbED-MT-history-second-turn"
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        for history_item in "${history_items[@]}"; do
            for history_window in "${history_windows[@]}"; do
                for w_grid_measure in "${history_w_grid_measures[@]}"; do
                    run_experiment "$exp_title" "TabedMT" \
                        "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy" \
                        "history_dependent=True" "history_item=$history_item" \
                        "history_window=$history_window" "history_w_grid_measure=$w_grid_measure"
                done
            done
        done
    done
done
run_evaluation "$exp_title" "EvalTabedMT" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}" \
    "eval_history_dependent=True" "eval_history_item=$(IFS=,; echo "${history_items[*]}")" \
    "eval_history_window=${history_windows[*]}" \
    "eval_history_w_grid_measure=${history_w_grid_measures[*]}"

# MTC: Multimodal + Text-only + Caption
exp_title="llava15-68m-7b-IbED-MTC-second-turn"
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        run_experiment "$exp_title" "TabedMTC" \
            "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy"
    done
done
run_evaluation "$exp_title" "EvalTabedMTC" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}" \
    "eval_captioning_model=${captioning_models[*]}" "eval_caption_type=${caption_types[*]}"

# MTC*: MTC with history
exp_title="llava15-68m-7b-IbED-MTC-history-second-turn"
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        for history_item in "${history_items[@]}"; do
            for history_window in "${history_windows[@]}"; do
                for w_grid_measure in "${history_w_grid_measures[@]}"; do
                    run_experiment "$exp_title" "TabedMTC" \
                        "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy" \
                        "history_dependent=True" "history_item=$history_item" \
                        "history_window=$history_window" "history_w_grid_measure=$w_grid_measure"
                done
            done
        done
    done
done
run_evaluation "$exp_title" "EvalTabedMTC" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}" \
    "eval_captioning_model=${captioning_models[*]}" "eval_caption_type=${caption_types[*]}" \
    "eval_history_dependent=True" "eval_history_item=$(IFS=,; echo "${history_items[*]}")" \
    "eval_history_window=${history_windows[*]}" \
    "eval_history_w_grid_measure=${history_w_grid_measures[*]}"

# MTP: Multimodal + Text-only + Pool
exp_title="llava15-68m-7b-IbED-MTP-second-turn"
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        for target_dim in "${target_dim_image_poolings[@]}"; do
            run_experiment "$exp_title" "TabedMTP" \
                "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy" \
                "target_dim_image_pooling=$target_dim"
        done
    done
done
run_evaluation "$exp_title" "EvalTabedMTP" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}" \
    "eval_target_dim_image_pooling=${target_dim_image_poolings[*]}"

# MTP*: MTP with history
exp_title="llava15-68m-7b-IbED-MTP-history-second-turn"
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        for target_dim in "${target_dim_image_poolings[@]}"; do
            for history_item in "${history_items[@]}"; do
                for history_window in "${history_windows[@]}"; do
                    for w_grid_measure in "${history_w_grid_measures[@]}"; do
                        run_experiment "$exp_title" "TabedMTP" \
                            "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy" \
                            "target_dim_image_pooling=$target_dim" \
                            "history_dependent=True" "history_item=$history_item" \
                            "history_window=$history_window" "history_w_grid_measure=$w_grid_measure"
                    done
                done
            done
        done
    done
done
run_evaluation "$exp_title" "EvalTabedMTP" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}" \
    "eval_target_dim_image_pooling=${target_dim_image_poolings[*]}" \
    "eval_history_dependent=True" "eval_history_item=$(IFS=,; echo "${history_items[*]}")" \
    "eval_history_window=${history_windows[*]}" \
    "eval_history_w_grid_measure=${history_w_grid_measures[*]}"

# MTCP: Multimodal + Text-only + Caption + Pool
exp_title="llava15-68m-7b-IbED-MTCP-second-turn"
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        for caption_type in "${caption_types[@]}"; do
            for captioning_model in "${captioning_models[@]}"; do
                for target_dim in "${target_dim_image_poolings[@]}"; do
                    run_experiment "$exp_title" "TabedMTCP" \
                        "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy" \
                        "caption_type=$caption_type" "captioning_model=$captioning_model" \
                        "target_dim_image_pooling=$target_dim"
                done
            done
        done
    done
done
run_evaluation "$exp_title" "EvalTabedMTCP" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}" \
    "eval_caption_type=${caption_types[*]}" "eval_captioning_model=${captioning_models[*]}" \
    "eval_target_dim_image_pooling=${target_dim_image_poolings[*]}"

# MTCP*: MTCP with history
exp_title="llava15-68m-7b-IbED-MTCP-history-second-turn"
history_items_mtcp=(kld w-grid)
for tabed_rule in "${tabed_rules[@]}"; do
    for mm_weight_policy in "${mm_weight_policys[@]}"; do
        for caption_type in "${caption_types[@]}"; do
            for captioning_model in "${captioning_models[@]}"; do
                for target_dim in "${target_dim_image_poolings[@]}"; do
                    for history_item in "${history_items_mtcp[@]}"; do
                        for history_window in "${history_windows[@]}"; do
                            for w_grid_measure in "${history_w_grid_measures[@]}"; do
                                run_experiment "$exp_title" "TabedMTCP" \
                                    "tabed_rule=$tabed_rule" "mm_weight_policy=$mm_weight_policy" \
                                    "caption_type=$caption_type" "captioning_model=$captioning_model" \
                                    "target_dim_image_pooling=$target_dim" \
                                    "history_dependent=True" "history_item=$history_item" \
                                    "history_window=$history_window" "history_w_grid_measure=$w_grid_measure"
                            done
                        done
                    done
                done
            done
        done
    done
done
run_evaluation "$exp_title" "EvalTabedMTCP" \
    "eval_tabed_rule=${tabed_rules[*]}" "eval_mm_weight_policy=${mm_weight_policys[*]}" \
    "eval_caption_type=${caption_types[*]}" "eval_captioning_model=${captioning_models[*]}" \
    "eval_target_dim_image_pooling=${target_dim_image_poolings[*]}" \
    "eval_history_dependent=True" "eval_history_item=$(IFS=,; echo "${history_items_mtcp[*]}")" \
    "eval_history_window=${history_windows[*]}" \
    "eval_history_w_grid_measure=${history_w_grid_measures[*]}"
