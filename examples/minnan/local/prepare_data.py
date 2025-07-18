# 适用于文本全在一个txt文件的

import argparse
import logging
import glob
import os
from tqdm import tqdm


logger = logging.getLogger()


def main():
    if 'vivos' in args.src_dir:
        wavs = list(glob.glob('{}/*/*/*wav'.format(args.src_dir)))
    else:
        wavs = list(glob.glob('{}/*/*wav'.format(args.src_dir)))
    
    # 先读入全部文本内容到字典
    txt = os.path.join(args.src_dir, 'text.txt')
    if not os.path.exists(txt):
        logger.warning('{} does not exist, try load prompts.txt'.format(txt))
        txt = os.path.join(args.src_dir, 'prompts.txt')
    utt2content = {}
    with open(txt, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                if 'vivos' in args.src_dir:
                    utt2content[utt_id] = ''.join('<|vi|>'+text)
                else:
                    utt2content[utt_id] = ''.join('<|min|>'+text)
                    
    # 初始化映射表
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in tqdm(wavs):
        utt = os.path.basename(wav).replace('.wav', '') # 语音id
        if 'vivos' in args.src_dir:
            spk = utt.split('_')[0] # 说话人id
        else:
            parts = utt.split('-')
            if len(parts) == 2:
                spk = parts[0]
            elif len(parts) == 3:
                spk = parts[0] + '-' + parts[1]
            else:
                raise Exception(f'意料之外的格式\nwav：{wav}\nutt：{utt}\n分割后：{parts}')
        # 查找对应文本内容
        if utt not in utt2content:
            logger.warning('Utterance {} not found in {}'.format(utt, txt))
            continue
        content = utt2content[utt]
                   
        utt2wav[utt] = wav # 语音->路径
        utt2text[utt] = content # 语音->内容
        utt2spk[utt] = spk # 语音->说话人       
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt) # 说话人->语音

    # 写出文件
    os.makedirs(args.des_dir, exist_ok=True)
    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()