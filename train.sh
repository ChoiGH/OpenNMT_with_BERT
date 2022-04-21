#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python3 train.py \
        -data ../data/bert_embedding/demo/kobert_bpe/kobert_demo \
        -save_model ../data/bert_embedding/model/transformer_bertenc_kobertdec_embedding_nmt_model/transformer_bertenc_kobertdec_embedding_model \
        -world_size 1 -gpu_ranks 0 \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 10000 \
        -pre_word_vecs_enc ../data/bert_embedding/model/embedding_file/kobertsize_kobert/size_of_kobert_embedding_en2ko_for_opennmt_gloves.enc.pt \
        -pre_word_vecs_dec ../data/bert_embedding/model/embedding_file/kobertsize_kobert/size_of_kobert_embedding_en2ko_for_opennmt_gloves.dec.pt

CUDA_VISIBLE_DEVICES=1 python3 train.py \
        -data ../data/bert_embedding/demo/bert/bert_demo \
        -save_model ../data/bert_embedding/model/transformer_seq_bert_embedding_nmt_model/transformer_seq_bert_embedding_model \
        -world_size 1 -gpu_ranks 0 \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 10000 \
        -pre_word_vecs_enc ../data/bert_embedding/model/embedding_file/seq_bert_emb/seq_size_of_bert_embedding_en2ko_for_opennmt_gloves.enc.pt \
        -pre_word_vecs_dec ../data/bert_embedding/model/embedding_file/seq_bert_emb/seq_size_of_bert_embedding_en2ko_for_opennmt_gloves.enc.pt