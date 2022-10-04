from copy import deepcopy
import random

import torch


class OscarInputs(object):
    def __init__(self, tokenizer, device="cpu", max_img_seq_length=50, max_seq_length=70,
                 max_seq_a_length=20, mask_prob=0.15, max_masked_tokens=3,
                 is_train=False, add_od_labels=False, input_params=None):

        """

        Parameters
        ----------
        tokenizer
        device
        max_img_seq_length
        max_seq_length
        max_seq_a_length
        mask_prob
        max_masked_tokens
        is_train
        input_params
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
                                                     self.max_seq_len), dtype=torch.long))

        self.device = device

        cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token, self.tokenizer.sep_token,
                                                  self.tokenizer.pad_token, self.tokenizer.mask_token, '.'])

        self.inputs_param = {'is_decode': True,
                             'do_sample': False,
                             'bos_token_id': cls_token_id,
                             'pad_token_id': pad_token_id,
                             'eos_token_ids': [sep_token_id],
                             'mask_token_id': mask_token_id,
                             # for adding od labels
                             'od_labels_start_posid': self.max_seq_a_len,

                             # hyperparameters of beam search
                             'max_length': self.max_seq_len,
                             'num_beams': 5,
                             "temperature": 1,
                             "top_k": 0,
                             "top_p": 1,
                             "repetition_penalty": 1,
                             "length_penalty": 1,
                             "num_return_sequences": 1,
                             "num_keep_best": 1,
                             }

        if input_params:
            self.inputs_param.update(input_params)

    def decode(self, outputs):

        all_caps = outputs[0].cpu()  # batch_size * num_keep_best * max_len
        all_confs = torch.exp(outputs[1].cpu())

        output_dict = {}
        for i, (caps, confs) in enumerate(zip(all_caps, all_confs)):
            res = []
            for cap, conf in zip(caps, confs):
                cap = self.tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                res.append({'caption': cap, 'conf': conf.item()})
            output_dict[i] = res

        return output_dict

    def encode(self, img_feat, caption="", labels=None,
               cls_token_segment_id=0, pad_token_segment_id=0,
               sequence_a_segment_id=0, sequence_b_segment_id=1, use_cbs=False):

        """

        Parameters
        ----------

        img_feat: torch.Tensor with shape (batch_size, num_feat, feat_size)
        caption
        labels
        cls_token_segment_id
        pad_token_segment_id
        sequence_a_segment_id
        sequence_b_segment_id
        use_cbs

        Returns
        -------

        """

        input_ids, attention_mask, token_type_ids, img_feats, masked_pos = [], [], [], [], []
        if len(img_feat.shape) == 3:
            # assuming batch_size is the first dim
            for i in range(img_feat.shape[0]):
                text_b = None
                if labels:
                    text_b = labels[i]

                tensors = self._tensorize(img_feat[i], text_a=caption, text_b=text_b,
                                          cls_token_segment_id=cls_token_segment_id,
                                          pad_token_segment_id=pad_token_segment_id,
                                          sequence_a_segment_id=sequence_a_segment_id,
                                          sequence_b_segment_id=sequence_b_segment_id)

                tensors = [t.unsqueeze(0) for t in tensors if len(t.shape) < 3]

                input_ids.append(tensors[0])
                attention_mask.append(tensors[1])
                token_type_ids.append(tensors[2])
                img_feats.append(tensors[3])
                masked_pos.append(tensors[4])

        else:
            raise ValueError("image_feat.shape == {} != 3, "
                             "Expected 3 dims: (batch_size, num_feat, feat_size)".format(img_feat.shape))

        self.inputs_param.update({'input_ids': torch.cat(input_ids), 'attention_mask': torch.cat(attention_mask),
                                  'token_type_ids': torch.cat(token_type_ids), 'img_feats': torch.cat(img_feats),
                                  'masked_pos': torch.cat(masked_pos)})

        if use_cbs:
            raise NotImplemented("cbs not implemented")

        return deepcopy(self.inputs_param)

    def _tensorize(self, img_feat, text_a="", text_b=None,
                   cls_token_segment_id=0, pad_token_segment_id=0,
                   sequence_a_segment_id=0, sequence_b_segment_id=1):

        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)

        if text_b:
            # format labels
            if isinstance(text_b, list):
                text_b = " ".join(text_b)
            else:
                text_b = text_b.replace(",", " ")
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int, device=self.device)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len))  # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                                               (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int, device=self.device)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]), device=self.device)
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long, device=self.device)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start: c_end, c_start: c_end].copy_(self._triangle_mask[0: seq_a_len, 0: seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        # full attention for L-R:
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=self.device)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long, device=self.device)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids)

        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos)
