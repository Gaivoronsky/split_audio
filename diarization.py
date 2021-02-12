from resemblyzer.voice_encoder import VoiceEncoder
from resemblyzer.hparams import sampling_rate
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import argparse
import os


def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="audio fragment")
    parser.add_argument("--noise", help="audio fragment with noise")
    parser.add_argument("--voice", help="audio fragment with voice")
    parser.add_argument("--plot", help="display image", action="store_true")
    parser.add_argument("--min_len_audio", help="minimal length split audio", default=0.2, type=float)
    parser.add_argument("--add_time", help="adding time for voice", default=0.3, type=float)
    parser.add_argument("--device", help="device for calculations", default='cpu')
    parser.add_argument("--dir_save", help="directory for save files", default='data')
    return parser


def convert_wav(path: str):
    """
        new_path = convert_wav(path)
        output: новое имя файла
    """
    new_path = path.replace(".wav", "_16.wav")
    os.system(f'ffmpeg -i {path} -loglevel quiet -acodec pcm_s16le -ac 1 -ar 16000 {new_path}')
    os.remove(path)
    os.rename(new_path, path)
    return path


def read_file(path):
    """
        data, samplerate = read_file('test.wav')
        output: data - массив значений аудио, samplerate - характеристика аудио
    """
    data, samplerate = sf.read(path)
    if samplerate != 16000:
        data, _ = read_file(convert_wav(path))
    return data, samplerate


def get_similarity(encoder, cont_embeds, speaker_wav):
    speaker_embeds = encoder.embed_utterance(speaker_wav)
    return cont_embeds @ speaker_embeds


def get_change_moments(similarity_dict, wav_splits, plot):
    prev_name = ''
    res = []
    res_names = []
    list_names = list(similarity_dict.keys())
    for i in range(len(wav_splits)):
        similarities = [s[i] for s in similarity_dict.values()]
        # print(f'{similarities[0]} {similarities[1]}')
        best = np.argmax(similarities)
        name, similarity = list_names[best], similarities[best]
        if name != prev_name:
            res.append(i)
            res_names.append(name)
        prev_name = name
    if plot:
        plt.figure()
        data_0 = similarity_dict[list_names[0]]
        data_1 = similarity_dict[list_names[1]]
        plt.plot(data_0)
        plt.plot(data_1)
        plt.show()
    diar_frags = list(wav_splits[res[1:]])
    choose = lambda x: x[0] + (x[1] - x[0]) / 3 # TODO
    return [wav_splits[0][0]] + list(map(choose, diar_frags)) + [wav_splits[-1][1]], res_names


def get_fragment_parts(change_moments, names):
    res = []
    for i in range(len(names)):
        res.append([names[i], change_moments[i], change_moments[i + 1]])
    return res


def split_wav(wav, sr, timeline, add_time, min_len_audio, dir_save):
    for idx, (label, start, end) in enumerate(timeline):
        if min_len_audio < end - start:
            if label == 'voice':
                sf.write(f'{dir_save}/{idx}-{label}.wav', wav[int(start * sr): int(end * sr) + int(add_time * sr)], sr)


def diarize(args):
    if not os.path.exists(f'./{args.dir_save}/'): os.mkdir(f'./{args.dir}/')

    encoder = VoiceEncoder(args.device, verbose=False)

    wav, sr_0 = read_file(args.audio)
    noise, sr_1 = read_file(args.noise)
    voice, sr_2 = read_file(args.voice)

    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    noise_similarity = get_similarity(encoder, cont_embeds, noise)
    voice_similarity = get_similarity(encoder, cont_embeds, voice)

    similarity_dict = {'noise': noise_similarity, 'voice': voice_similarity}
    wav_splits_seconds = np.array(list(map(lambda x: [x.start / sampling_rate, x.stop / sampling_rate], wav_splits)))

    change_moments, names = get_change_moments(similarity_dict, wav_splits_seconds, args.plot)
    diarized_fragments = get_fragment_parts(change_moments, names)
    split_wav(wav, sr_0, diarized_fragments, dir_save=args.dir_save, add_time=args.add_time, min_len_audio=args.min_len_audio)

    return diarized_fragments


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    diarize(args)