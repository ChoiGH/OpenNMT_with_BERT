#!/bin/sh
python3 translate.py -model ../data/bert_embedding/model/transformer_bert_embedding_only_encoder/only_encoder_model_step_200000.pt \
 -src ../data/test/biz_test_2009_11_3000sent/biz.test.en_bert_base_cased.token.txt \
 -output ../data/test/biz_test_2009_11_3000sent/biz.test.en_bert_base_cased.token.txt.only_encoder_bert.out \
 -gpu 0

python3 translate.py -model ../data/bert_embedding/model/transformer_bert_embedding_only_encoder/only_encoder_model_step_200000.pt \
 -src ../data/test/lecture_test_2019_03_2290sent/lecture.test.en_bert_base_cased.token.txt \
 -output ../data/test/lecture_test_2019_03_2290sent/lecture.test.en_bert_base_cased.token.txt.only_encoder_bert.out \
 -gpu 0

python3 translate.py -model ../data/bert_embedding/model/transformer_bert_embedding_only_encoder/only_encoder_model_step_200000.pt \
 -src ../data/test/it_test/it_web_2000.txt_google_bert_token.txt \
 -output ../data/test/it_test/it_web_2000.txt_google_bert_token.txt.only_encoder_bert.out \
 -gpu 0

python3 translate.py -model ../data/bert_embedding/model/transformer_bert_embedding_only_encoder/only_encoder_model_step_200000.pt \
 -src ../data/test/Trip_test/trip.test.en_bert_base_cased.token.txt \
 -output ../data/test/Trip_test/trip.test.en_bert_base_cased.token.txt.only_encoder_bert.out \
 -gpu 0

python3 translate.py -model ../data/bert_embedding/model/transformer_bert_embedding_only_encoder/only_encoder_model_step_200000.pt \
 -src ../data/test/Trip_test/trip2.test.en_bert_base_cased.token.txt \
 -output ../data/test/Trip_test/trip2.test.en_bert_base_cased.token.txt.only_encoder_bert.out \
 -gpu 0