
# Training and Inference on the MLB Dataset

```bash
SEQ_PLAN=<path_to_seq_plan_repo>
```

Do not forget to set the correct branch:

```bash
cd $SEQ_PLAN
git checkout main
```

## __I.__ Training the model
1. Process the data

```bash
MLB=$SEQ_PLAN/mlb/
mkdir $MLB
ORDINAL_ADJECTIVE_MAP_FOLDER=$SEQ_PLAN/data
cd $SEQ_PLAN/scripts
python create_mlb_target_data.py -output_folder $MLB -dataset_type train -ordinal_adjective_map_file \
${ORDINAL_ADJECTIVE_MAP_FOLDER}/traintokens-ordinaladjective-inning-identifier
python create_mlb_target_data.py -output_folder $MLB -dataset_type valid -ordinal_adjective_map_file \
${ORDINAL_ADJECTIVE_MAP_FOLDER}/validtokens-ordinaladjective-inning-identifier
python create_mlb_target_data.py -output_folder $MLB -dataset_type test -ordinal_adjective_map_file \
${ORDINAL_ADJECTIVE_MAP_FOLDER}/testtokens-ordinaladjective-inning-identifier
```
2: Run bpe tokenization
```
MERGES=16000
TRAIN_FILE_1=$MLB/train.pp
TRAIN_FILE_2=$MLB/train.su
COMBINED=$MLB/combined

MLB_TOKENIZED=$SEQ_PLAN/mlb_tokenized
mkdir $MLB_TOKENIZED

CODE=$MLB_TOKENIZED/code
cat $TRAIN_FILE_1 $TRAIN_FILE_2 > $COMBINED

cd $SEQ_PLAN/tools
python learn_bpe.py -s 6000 < $COMBINED > $CODE 

TRAIN_BPE_FILE_1=$MLB_TOKENIZED/train.bpe.pp
TRAIN_BPE_FILE_2=$MLB_TOKENIZED/train.bpe.su

python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $TRAIN_FILE_1 > $TRAIN_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $TRAIN_FILE_2 > $TRAIN_BPE_FILE_2


VALID_FILE_1=$MLB/valid.pp
VALID_FILE_2=$MLB/valid.su
VALID_BPE_FILE_1=$MLB_TOKENIZED/valid.bpe.pp
VALID_BPE_FILE_2=$MLB_TOKENIZED/valid.bpe.su
python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $VALID_FILE_1 > $VALID_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $VALID_FILE_2 > $VALID_BPE_FILE_2
```

3. Preprocess the dataset
```
PREPROCESS=$SEQ_PLAN/preprocess
mkdir $PREPROCESS

IDENTIFIER=mlb

cd $SEQ_PLAN
python preprocess.py -train_src $MLB_TOKENIZED/train.bpe.pp \
                     -train_tgt $MLB_TOKENIZED/train.bpe.su \
                     -train_tgt_chosen_pp $MLB/train.macroplan \                     
                     -valid_src $MLB_TOKENIZED/valid.bpe.pp \
                     -valid_tgt $MLB_TOKENIZED/valid.bpe.su \             
                     -valid_tgt_chosen_pp $MLB/valid.macroplan\
                     -save_data $BASE/$PREPROCESS/mlb \
                     -src_seq_length 1000000 \
                     -tgt_seq_length 1000000 \
                     -shard_size 1024\
                     -dynamic_dict
```
4. Train model 
```
MODELS=$SEQ_PLAN/models
mkdir $MODELS
mkdir $MODELS/$IDENTIFIER

cd $SEQ_PLAN
python train.py -data $PREPROCESS/$IDENTIFIER \
                -save_model $MODELS/$IDENTIFIER/model \
                -encoder_type classifier \
                -layers 1 \
                -word_vec_size 300 \
                -rnn_size 700 \
                -seed 1234 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -report_every 10 \
                -batch_size 2 \
                -valid_batch_size 2 \
                -copy_attn \
                -reuse_copy_attn \
                -train_steps 100000 \
                -valid_steps 400 \
                -save_checkpoint_steps 400 \
                -start_decay_steps 100000 \
                -decay_steps 680 \
                --early_stopping 10 \
                --early_stopping_criteria accuracy \
                -world_size 4 \
                -gpu_ranks 0 1 2 3 \
                -gumbel_softmax_temp 0.1 \
                -pp_prediction_loss_multiplier 2 \
                -min_teacher_forcing_ratio 0 \
                -kl_loss_multiplier 1 \
                -accum_count 1 \
                -max_training_steps 100000 \
                --max_generator_batches=0 \
                --keep_checkpoint 15
```
## __II.__ Construct input of model during inference
1. Create the inference input
```
DATASET_TYPE=<one_of_{valid|test}>
SUFFIX=infer

cd $MACRO_PLAN/scripts
python construct_inference_mlb_input.py  -output_folder $MLB \
                                        -dataset_type $DATASET_TYPE \
                                        -suffix $SUFFIX
```
2. Apply bpe
```
CODE=$MLB_TOKENIZED/code  
FILE_1=$MLB/$DATASET_TYPE.$SUFFIX.pp
BPE_FILENAME=$MLB_TOKENIZED/$DATASET_TYPE.bpe.$SUFFIX.pp

cd $MACRO_PLAN/tools
python apply_bpe.py  -c $CODE --vocabulary-threshold 10 -suffix $SUFFIX <$FILE_1 >$BPE_FILENAME
```

## __III.__ Generate a summary
1. Run inference for the model
```
MODEL_PATH=$MODELS/$IDENTIFIER/<best_checkpoint>
GEN=$SEQ_PLAN/gen
mkdir $GEN

cd $SEQ_PLAN
python translate.py -model $MODEL_PATH \
                    -src $BPE_FILENAME \
                    -output $GEN/$IDENTIFIER_step-bpe_beam5_gens.txt \
                    -batch_size 10 \
                    -max_length 200 \
                    -gpu 0 \
                    -min_length 10 \
                    -pp_output $GEN/$IDENTIFIER_pp_out_gens.txt \
                    -beam_size 5
```
```
MODEL_PATH=$MODELS/$IDENTIFIER/<best_checkpoint>
GEN=$SEQ_PLAN/gen
mkdir $GEN

cd $SEQ_PLAN
python post_processing_step_output.py -step_file $GEN/$IDENTIFIER_step-bpe_beam5_gens.txt \
                    -summary_file $GEN/$IDENTIFIER-bpe_beam5_gens.txt \
                    -pp_out_file $GEN/$IDENTIFIER_pp_out_gens.txt \
                    -pp_inp_file $GEN/$IDENTIFIER_pp_inp_gens.txt 
```
```
for i in {1..20}  # run for subsequent steps  
do
python translate.py -model $MODEL_PATH \
                    -src $BPE_FILENAME \
                    -tgt $GEN/$IDENTIFIER-bpe_beam5_gens.txt \
                    -output $GEN/$IDENTIFIER_step-bpe_beam5_gens.txt \
                    -batch_size 10 \
                    -max_length 200 \
                    -gpu 0 \
                    -min_length 10 \
                    -block_ngram_plan_repeat 2 \
                    -block_repetitions 3  \
                    -current_para ${i} \
                    -min_para 12 \
                    -block_consecutive_repetitions 1 \
                    -pp_input $GEN/$IDENTIFIER_pp_inp_gens.txt \
                    -pp_output $GEN/$IDENTIFIER_pp_out_gens.txt \
                    -beam_size 5
cd $SEQ_PLAN
python post_processing_step_output.py -step_file $GEN/$IDENTIFIER_step-bpe_beam5_gens.txt \
                    -summary_file $GEN/$IDENTIFIER-bpe_beam5_gens.txt \
                    -pp_out_file $GEN/$IDENTIFIER_pp_out_gens.txt \
                    -pp_inp_file $GEN/$IDENTIFIER_pp_inp_gens.txt                     
```

2. Strip the ```@@@``` characters.
```
rm $GEN/$IDENTIFIER_step-bpe_beam5_gens.txt
sed 's/ <segment> <end-summary>.*$//g' $GEN/$IDENTIFIER-bpe_beam5_gens.txt \  
  >$GEN/$IDENTIFIER-bpe_strip_beam5_gens.txt  
  
sed -r 's/(@@ )|(@@ ?$)//g; s/<segment> //g' $GEN/$IDENTIFIER-bpe_strip_beam5_gens.txt \  
  >$GEN/$IDENTIFIER-beam5_gens.txt

```


## Evaluation
Note: The IE evaluation has a preprocessing step of identifying mentions of innings. For this, we make use of GPT-2 to check if an ordinal adjective is an inning or not. The details are mentioned in the paper. You can install the required version of HuggingFace Transformers library using the below command.
```
pip install transformers==2.3
```
The detailed evaluation steps are below:

1. Compute BLEU
```
REFERENCE=$MLB/test.strip.su  # contains reference without <segment> tags
perl ~/mosesdecoder/scripts/generic/multi-bleu.perl  $REFERENCE < $GENS/$IDENTIFIER-beam5_gens.txt
```

2. Preprocessing for IE

```
cd $SEQ_PLAN/scripts
```
```
python add_segment_marker.py -input_file $GEN/$IDENTIFIER-beam5_gens.txt -output_file \  
$GEN/$IDENTIFIER-segment-beam5_gens.txt
```
```
python inning_prediction_offline.py -input_file $GEN/$IDENTIFIER-segment-beam5_gens.txt \
-output_file $GEN/$IDENTIFIER-inning-map-beam5_gens.txt
```

```
IE_ROOT=~/ie_root
python mlb_data_utils.py -mode prep_gen_data -gen_fi $GEN/$IDENTIFIER-segment-beam5_gens.txt \
-dict_pfx "$IE_ROOT/data/mlb-ie" -output_fi $SEQ_PLAN/transform_gen/$IDENTIFIER-beam5_gens.h5 \
-input_path "$IE_ROOT/json" \
-ordinal_inning_map_file $GEN/$IDENTIFIER-inning-map-beam5_gens.txt \
-test
```
3. Run RG evaluation
```
IE_ROOT=~/ie_root
th extractor.lua -gpuid  0 -datafile $IE_ROOT/data/mlb-ie.h5 \
-preddata $SEQ_PLAN/transform_gen/$IDENTIFIER-beam5_gens.h5 -dict_pfx \
"$IE_ROOT/data/mlb-ie" -just_eval -ignore_idx 14 -test
```
4. Run evaluation for non rg metrics 
```
python non_rg_metrics.py $SEQ_PLAN/transform_gen/test_inn_mlb-beam5_gens.h5-tuples.txt \
$SEQ_PLAN/transform_gen/$IDENTIFIER-beam5_gens.h5-tuples.txt
```

