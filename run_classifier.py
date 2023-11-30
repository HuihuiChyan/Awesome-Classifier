import os
import xlrd
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertModel, BertPreTrainedModel, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, is_train=True):
        self.tokenizer = tokenizer
        self.data = data
        self.is_train = is_train
    def __getitem__(self, index):
        # 策略一：1比1截断
        # if self.is_train:
        #     x = self.tokenizer.encode_plus(self.data[index][0] + "[SEP]" + self.data[index][1],\
        #                                    self.data[index][2] + "[SEP]" + self.data[index][3],\
        #                                    padding='max_length',\
        #                                    max_length=512,\
        #                                    truncation=True)
            
        #     return x['input_ids'], x['token_type_ids'], x['attention_mask'], self.data[index][4]
        # else:
        #     x = self.tokenizer.encode_plus(self.data[index][0] + "[SEP]" + self.data[index][1],\
        #                                    self.data[index][2] + "[SEP]" + self.data[index][3],\
        #                                    padding='max_length',\
        #                                    max_length=512,\
        #                                    truncation=True)

        #     return x['input_ids'], x['token_type_ids'], x['attention_mask']

        # 策略二：1比4截断
        # tokens1 = self.tokenizer.tokenize(self.data[index][0] + "[SEP]" + self.data[index][1])
        # tokens2 = self.tokenizer.tokenize(self.data[index][2] + "[SEP]" + self.data[index][3])
        # indices1 = self.tokenizer.convert_tokens_to_ids(tokens1)
        # indices2 = self.tokenizer.convert_tokens_to_ids(tokens2)
        
        # if len(indices1) + len(indices2) >= 510:
        #     indices1 = indices1[:110]
        #     indices2 = indices2[:400]

        # x = self.tokenizer.prepare_for_model(indices1,
        #                                      indices2,
        #                                      padding="max_length",
        #                                      max_length=512,
        #                                      truncation="only_second",
        #                                      return_overflowing_tokens=False)
        # if self.is_train:
        #     return x['input_ids'], x['token_type_ids'], x['attention_mask'], self.data[index][4]
        # else:
        #     return x['input_ids'], x['token_type_ids'], x['attention_mask']

        # 策略三：优先截断reply
        tokens1 = self.tokenizer.tokenize(self.data[index][0] + "[SEP]" + self.data[index][1])
        tokens2 = self.tokenizer.tokenize(self.data[index][2] + "[SEP]" + self.data[index][3])
        indices1 = self.tokenizer.convert_tokens_to_ids(tokens1)
        indices2 = self.tokenizer.convert_tokens_to_ids(tokens2)
        
        if len(indices2) < 480:
            x = self.tokenizer.prepare_for_model(indices1,
                                                 indices2,
                                                 padding="max_length",
                                                 max_length=500,
                                                 truncation="only_first",
                                                 return_overflowing_tokens=False)
        else:
            x = self.tokenizer.prepare_for_model(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.data[index][0])),
                                                 indices2,
                                                 padding="max_length",
                                                 max_length=500,
                                                 truncation="only_second",
                                                 return_overflowing_tokens=False)
        if self.is_train:
            return x['input_ids'], x['token_type_ids'], x['attention_mask'], self.data[index][4], self.data[index][5], self.data[index][6], self.data[index][7], self.data[index][8]
        else:
            return x['input_ids'], x['token_type_ids'], x['attention_mask']

        # # 策略四：优先截断reply，并且对liuyan只截断中间的部分
        # tokens1 = self.tokenizer.tokenize(self.data[index][0] + "[SEP]" + self.data[index][1])
        # tokens2 = self.tokenizer.tokenize(self.data[index][2] + "[SEP]" + self.data[index][3])
        # indices1 = self.tokenizer.convert_tokens_to_ids(tokens1)
        # indices2 = self.tokenizer.convert_tokens_to_ids(tokens2)
        
        # if len(indices2) < 480:
        #     x = self.tokenizer.prepare_for_model(indices1,
        #                                          indices2,
        #                                          padding="max_length",
        #                                          max_length=500,
        #                                          truncation="only_first",
        #                                          return_overflowing_tokens=False)
        # else:
        #     indices2 = indices2[:240] + indices2[-240:]
        #     x = self.tokenizer.prepare_for_model(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.data[index][0])),
        #                                          indices2,
        #                                          padding="max_length",
        #                                          max_length=500,
        #                                          truncation="only_first",
        #                                          return_overflowing_tokens=False)
        # if self.is_train:
        #     return x['input_ids'], x['token_type_ids'], x['attention_mask'], self.data[index][4], self.data[index][5], self.data[index][6], self.data[index][7], self.data[index][8]
        # else:
        #     return x['input_ids'], x['token_type_ids'], x['attention_mask']

    def __len__(self):
        return len(self.data)

def train(args, train_dataloader, valid_dataloader, model, tokenizer):
    
    best_f1=0.0
    global_step=0
    best_step=0

    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, args.loss_weight]).cuda())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.max_epoch):
        for inputs_ids, token_type_ids, attention_mask, tag1, tag2, tag3, tag4, tag_cp in tqdm(train_dataloader, desc="Epoch "+str(epoch)):
            model.train()
            inputs_ids=torch.LongTensor([i.tolist() for i in inputs_ids])
            token_type_ids=torch.LongTensor([i.tolist() for i in token_type_ids])
            attention_mask=torch.LongTensor([i.tolist() for i in attention_mask])
            output=model(inputs_ids.cuda().transpose(0,1), token_type_ids.cuda().transpose(0,1), attention_mask.cuda().transpose(0,1))
            
            loss1=criterion(output[0], tag1.cuda())
            loss2=criterion(output[1], tag2.cuda())
            loss3=criterion(output[2], tag3.cuda())
            loss4=criterion(output[3], tag4.cuda())
            loss_cp=criterion(output[4], tag_cp.cuda())
            loss=loss1 + loss2 + loss3 + loss4 + loss_cp
            # loss=loss5
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step=global_step+1
            
            if global_step%50==0:
                print('\nEpoch: %d, step: %d, loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4: %.4f, loss_cp: %.4f.' % (epoch, global_step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss_cp.item()))
        
                all_preds_valid = [[], [], [], [], []]
                all_tags_valid = [[], [], [], [], []]
                for valid_inputs_ids, valid_token_type_ids, valid_attention_mask, valid_tag1, valid_tag2, valid_tag3, valid_tag4, valid_tag_cp in valid_dataloader:
                    model.eval()
                    valid_inputs_ids=torch.LongTensor([i.tolist() for i in valid_inputs_ids])
                    valid_token_type_ids=torch.LongTensor([i.tolist() for i in valid_token_type_ids])
                    valid_attention_mask=torch.LongTensor([i.tolist() for i in valid_attention_mask])
                    output=model(valid_inputs_ids.cuda().transpose(0,1), valid_token_type_ids.cuda().transpose(0,1), valid_attention_mask.cuda().transpose(0,1))
                    valid_tags = [valid_tag1, valid_tag2, valid_tag3, valid_tag4, valid_tag_cp]
                    for i, o in enumerate(output):
                        all_preds_valid[i].extend(torch.argmax(o, dim=-1).tolist())
                        all_tags_valid[i].extend(valid_tags[i].tolist())

        print('\nEpoch: %d, step: %d' % (epoch, global_step))

        f1_values = []
        for i in range(5):
            f1_value = f1_score(all_tags_valid[i], all_preds_valid[i])
            precision_value = precision_score(all_tags_valid[i], all_preds_valid[i])
            recall_value = recall_score(all_tags_valid[i], all_preds_valid[i])
            
            f1_values.append(f1_value)
            
            print('For class %d, valid_f1: %.4f, valid_precision: %.4f, valid_recall: %.4f.' % (i, f1_value, precision_value, recall_value))
        
        avg_f1_value = sum(f1_values)/5

        print('The averaged f1_value of 5 classes is %.4f' % (avg_f1_value))

        if avg_f1_value > best_f1:
            best_f1 = avg_f1_value
            best_step = global_step
            tokenizer.save_pretrained(args.output_path)
            model.save_pretrained(args.output_path)
            print('Best F1 on valid set! Now saving model to %s.' % (args.output_path))
        else:
            print('Not the best F1 on valid set. The best F1 is %.4f at step %d.' % (best_f1, best_step))

def infer(infer_dataloader, model, output_path):

    infer_preds = []

    for inputs_ids, token_type_ids, attention_mask in tqdm(infer_dataloader, desc="Infering"):
        model.eval()
        inputs_ids=torch.LongTensor([i.tolist() for i in inputs_ids])
        token_type_ids=torch.LongTensor([i.tolist() for i in token_type_ids])
        attention_mask=torch.LongTensor([i.tolist() for i in attention_mask])
        output=model(inputs_ids.cuda().transpose(0,1), token_type_ids.cuda().transpose(0,1), attention_mask.cuda().transpose(0,1))

        infer_preds1=torch.argmax(output[0], dim=-1).cpu().tolist()
        infer_preds2=torch.argmax(output[1], dim=-1).cpu().tolist()
        infer_preds3=torch.argmax(output[2], dim=-1).cpu().tolist()
        infer_preds4=torch.argmax(output[3], dim=-1).cpu().tolist()
        infer_preds_cp=torch.argmax(output[4], dim=-1).cpu().tolist()

        for i in range(len(infer_preds1)):

            infer_preds.append([infer_preds1[i], infer_preds2[i], infer_preds3[i], infer_preds4[i], infer_preds_cp[i]])

    return infer_preds

class BertForCoClassification(BertPreTrainedModel):
    def __init__(self, config, multi_task=False):
        super().__init__(config)
        self.bert=BertModel(config)
        self.dropout=torch.nn.Dropout(p=0.3)
        self.multi_task=multi_task
        if self.multi_task:
            self.fc1=torch.nn.Linear(config.hidden_size, 2)
            self.fc2=torch.nn.Linear(config.hidden_size, 2)
            self.fc3=torch.nn.Linear(config.hidden_size, 2)
            self.fc4=torch.nn.Linear(config.hidden_size, 2)
            self.fc_cp=torch.nn.Linear(config.hidden_size, 2)
        else:
            self.fc=torch.nn.Linear(config.hidden_size, 2)
        self.post_init()
    def forward(self,inputs_ids,token_type_ids,attention_mask):
        out = self.bert(input_ids=inputs_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
        out = out.last_hidden_state[:, 0]
        out = self.dropout(out)
        if self.multi_task:
            output1=self.fc1(out)
            output2=self.fc2(out)
            output3=self.fc3(out)
            output4=self.fc4(out)
            output_cp=self.fc_cp(out)
            return output1, output2, output3, output4, output_cp
        else:
            output=self.fc(out)
            return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-or-test", type=str, default="do-train", choices=("do-train", "do-test"))
    parser.add_argument("--model-path", type=str, default="./chinese-macbert-base")
    parser.add_argument("--output-path", type=str, default="./saved_best_model")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--learning-rate", type=int, default=1e-5)
    parser.add_argument("--loss-weight", type=int, default=1)
    args = parser.parse_args()

    ###############################################################################
    # 输入格式说明：四列，分别为title、liuyan、reply_type、reply                     #
    # 输出格式说明：五列，分别为coplan、codesign、codelivery、coassess、coproduction #
    ###############################################################################

    if args.train_or_test == "do-train":

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        # fetch the data from excel sheets!
        all_wb = xlrd.open_workbook("./data/processed_NL.xlsx")
        train_ws = all_wb.sheet_by_index(0)
        valid_ws = all_wb.sheet_by_index(1)
        train_data = []
        valid_data = []
        for i in range(1, train_ws.nrows):
            # 0 province, 1 year, 2 title, 3 liuyan, 4 reply_type, 5 reply, 6 coplan, 7 codesign, 8 codelivery, 9 coassess, 10 coproduction
            train_data.append([train_ws.cell_value(i, 2).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               train_ws.cell_value(i, 3).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               train_ws.cell_value(i, 4).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               train_ws.cell_value(i, 5).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               int(train_ws.cell_value(i, 6)==1.0),
                               int(train_ws.cell_value(i, 7)==1.0),
                               int(train_ws.cell_value(i, 8)==1.0),
                               int(train_ws.cell_value(i, 9)==1.0),
                               int(train_ws.cell_value(i, 10)==1.0)])

        for i in range(2, valid_ws.nrows):
            # 0 province, 1 year, 2 title, 3 liuyan, 4 reply_type, 5 reply, 6 coplan, 7 codesign, 8 codelivery, 9 coassess, 10 coproduction
            valid_data.append([valid_ws.cell_value(i, 2).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               valid_ws.cell_value(i, 3).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               valid_ws.cell_value(i, 4).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               valid_ws.cell_value(i, 5).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               int(valid_ws.cell_value(i, 6)==1.0),
                               int(valid_ws.cell_value(i, 7)==1.0),
                               int(valid_ws.cell_value(i, 8)==1.0),
                               int(valid_ws.cell_value(i, 9)==1.0),
                               int(valid_ws.cell_value(i, 10)==1.0)])
                            
        all_wb = xlrd.open_workbook("./data/20-22环保留言-train.xlsx")
        train_ws = all_wb.sheet_by_index(1) # the sheet of 3 Cos
        for i in range(1, train_ws.nrows):
            # 0 省, 1 市, 2 区县, 3 留言类型, 4 留言对象, 5 留言标题, 6 留言时间, 7 留言正文, 8 是否有图, 9 回复类型, 10 回复时间, 11 回复内容, 12 文章链接,
            # 13 留言用户id, 14 一次解决程度, 15 一次办理态度, 16 一次办理速度, 17 满意度, 18 pred, 19 coplan, 20 codesign, 21 codelivery, 22 coassess
            train_data.append([train_ws.cell_value(i, 5).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               train_ws.cell_value(i, 7).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               train_ws.cell_value(i, 9).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               train_ws.cell_value(i, 11).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", ""),
                               int(train_ws.cell_value(i, 19)==1.0),
                               int(train_ws.cell_value(i, 20)==1.0),
                               int(train_ws.cell_value(i, 21)==1.0),
                               int(train_ws.cell_value(i, 22)==1.0),
                               1]) # we think the data from 3 Cos are all coproduction

        # train_ws = all_wb.sheet_by_index(0) # to fetch some negative samples from 20-21 data
        # negative_count = 0
        # for i in range(1, train_ws.nrows):
        #     # 0 省, 1 市, 2 区县, 3 留言类型, 4 留言对象, 5 留言标题, 6 留言时间, 7 留言正文, 8 是否有图, 9 回复类型, 10 回复时间, 11 回复内容, 12 文章链接,
        #     # 13 留言用户id, 14 一次解决程度, 15 一次办理态度, 16 一次办理速度, 17 满意度, 18 pred, 19 coplan, 20 codesign, 21 codelivery, 22 coassess
        #     if train_ws.cell_value(i, 18) == 0:
        #         train_data.append([train_ws.cell_value(i, 5).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            train_ws.cell_value(i, 7).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            train_ws.cell_value(i, 9).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            train_ws.cell_value(i, 11).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            0, 0, 0, 0, 0]) # we think the data from negative samples are labeled as all 0
        #         negative_count += 1
        #     if negative_count >= 500:
        #         break # fetch 500 negative samples here

        # train_ws = all_wb.sheet_by_index(2) # to fetch some negative samples from 21-22 data
        # negative_count = 0
        # for i in range(1, train_ws.nrows):
        #     # 0 省, 1 市, 2 区县, 3 留言类型, 4 留言对象, 5 留言标题, 6 留言时间, 7 留言正文, 8 是否有图, 9 回复类型, 10 回复时间, 11 回复内容, 12 文章链接,
        #     # 13 留言用户id, 14 一次解决程度, 15 一次办理态度, 16 一次办理速度, 17 满意度, 18 pred, 19 coplan, 20 codesign, 21 codelivery, 22 coassess
        #     if train_ws.cell_value(i, 18) == 0:
        #         train_data.append([train_ws.cell_value(i, 5).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            train_ws.cell_value(i, 7).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            train_ws.cell_value(i, 9).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            train_ws.cell_value(i, 11).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
        #                            0, 0, 0, 0, 0]) # we think the data from negative samples are labeled as all 0
        #         negative_count += 1
        #     if negative_count >= 500:
        #         break # fetch 500 negative samples here
        
        all_wb = xlrd.open_workbook("./data/20-22环保留言-dev.xlsx")
        valid_ws = all_wb.sheet_by_index(0)
        for i in range(1, valid_ws.nrows):
            # 0 省, 1 市, 2 区县, 3 留言类型, 4 留言对象, 5 留言标题, 6 留言时间, 7 留言正文, 8 是否有图, 9 回复类型, 10 回复时间, 11 回复内容, 12 文章链接,
            # 13 留言用户id, 14 一次解决程度, 15 一次办理态度, 16 一次办理速度, 17 满意度, 18 pred, 19 coplan, 20 codesign, 21 codelivery, 22 coassess
            valid_data.append([train_ws.cell_value(i, 5).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
                               train_ws.cell_value(i, 7).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
                               train_ws.cell_value(i, 9).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
                               train_ws.cell_value(i, 11).strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", ""),
                               int(train_ws.cell_value(i, 19)==1.0),
                               int(train_ws.cell_value(i, 20)==1.0),
                               int(train_ws.cell_value(i, 21)==1.0),
                               int(train_ws.cell_value(i, 22)==1.0),
                               int(train_ws.cell_value(i, 19)==1.0) | int(train_ws.cell_value(i, 20)==1.0) | int(train_ws.cell_value(i, 21)==1.0) | int(train_ws.cell_value(i, 22)==1.0),]) 
                               # the coproduction can be induced by the other four cos

        # data fetch finished, now start build model and train!
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        train_dataset = MyDataset(tokenizer, train_data)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        valid_dataset = MyDataset(tokenizer, valid_data)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        model = BertForCoClassification.from_pretrained(args.model_path, multi_task=True).cuda()

        train(args, train_dataloader, valid_dataloader, model, tokenizer)

    else:
        import csv
        
        infer_data = []
        data_type_preds = []
        keywords = ["环境", "生态", "环境保护", "环境污染", "排放", "废水", "污水", "空气污染", "废气", "废气", "灰尘", "烟雾", "噪音", "辐射", "废渣", "垃圾", "病毒", "酸雨", "土壤"]
        with open("data/14-21cityfinal.csv", newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # 0 id, 1 province, 2 city, 3 year, 4 title, 5 liuyan_time, 6 liuyan_content, 7 liuyan_type, 8 reply_time, 9 reply_content, 10 isimage, 
            # 11 reply_type, 12 person, 13 url, 14 userid, 15 一次解决程度, 16 一次办理态度, 17 一次办理速度, 18 满意度
            for row in tqdm(csv_reader, desc="Processing"):
                assert len(row) == 19

                title = row[4].strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", "")
                liuyan = row[6].strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", "")
                reply_type = row[11].strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", "")
                reply_content = row[9].strip().replace("\xa0", "").replace("\u3000", "").replace("\\n", "").replace("\n", "")

                infer_data.append([title, liuyan, reply_type, reply_content])

                if title == "" or liuyan == "" or reply_type == "" or reply_content == "":
                    data_type_preds.append('Invalid')
                else:
                    sum1 = sum([int(word in title) for word in keywords])
                    sum2 = sum([int(word in liuyan) for word in keywords])
                    sum3 = sum([int(word in reply_content) for word in keywords])
                    if sum1 + sum2 + sum3 == 0:
                        data_type_preds.append("OODomain")
                    else:
                        data_type_preds.append("INDomain")
        
        tokenizer = AutoTokenizer.from_pretrained(args.output_path)

        infer_dataset = MyDataset(tokenizer, infer_data[1:], is_train=False)
        infer_dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)

        model = BertForCoClassification.from_pretrained(args.output_path, multi_task=True).cuda()
        infer_preds = infer(infer_dataloader, model, args.output_path)

        fout = open(os.path.join(args.output_path, "output.txt"), "w", encoding="utf-8")
        fout.write("data_type\tcoplan\tcodesign\tcodelivery\tcoassess\tcoproduction\n")
        for i in range(len(infer_data)-1):
            fout.write(data_type_preds[i+1]+"\t"+str(infer_preds[i][0])+"\t"+str(infer_preds[i][1])+"\t"+str(infer_preds[i][2])+"\t"+str(infer_preds[i][3])+"\t"+str(infer_preds[i][4])+"\n")

        print("Infer result written to %s." % os.path.join(args.output_path, "output.txt"))