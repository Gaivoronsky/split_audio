from resemblyzer.voice_encoder import VoiceEncoder
from resemblyzer.hparams import sampling_rate
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import argparse
import json
import os

from tqdm import tqdm
from pydub import AudioSegment
import math


def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="audio fragment or directory with them")
    parser.add_argument("--noise", help="audio fragment with noise")
    parser.add_argument("--voice", help="audio fragment with voice")
    parser.add_argument("--cuts_on", help="parts duration in seconds", default=None, type=int, required=False)
    parser.add_argument("--plot", help="display image", action="store_true")
    parser.add_argument("--min_len_audio", help="minimal length split audio", default=0.2, type=float)
    parser.add_argument("--add_time", help="adding time for voice", default=0.3, type=float)
    parser.add_argument("--device", help="device for calculations", default='cpu')
    parser.add_argument("--dir_save", help="directory for save files", default='data')
    parser.add_argument("--json", help="output file json", action="store_true")
    parser.add_argument("--save_split", help="output audio files", default=True, type=bool)
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
    choose = lambda x: x[0] + (x[1] - x[0]) / 3  # TODO
    return [wav_splits[0][0]] + list(map(choose, diar_frags)) + [wav_splits[-1][1]], res_names


def get_fragment_parts(change_moments, names, ):
    res = []
    for i in range(len(names)):
        res.append([names[i], change_moments[i], change_moments[i + 1]])
    return res


def split_wav(wav, sr, timeline, add_time, min_len_audio, dir_save, start_from=None):
    voise_iterator = tqdm(timeline, ascii=True, leave=False)
    for idx, (label, start, end) in enumerate(voise_iterator):
        voise_iterator.set_description(f'{label} fragment {idx} processing', refresh=True)
        if min_len_audio < end - start:
            if label == 'voice':
                if start_from:
                    voice_file_name = f'{dir_save}/{idx + start_from}-{label}'
                else:
                    voice_file_name = f'{dir_save}/{idx}-{label}'
                if os.path.exists(voice_file_name):
                    print(f'File "{voice_file_name}" already existed!')
                    voice_file_name = voice_file_name + '_1'

                voice_file_name = f'{voice_file_name}.wav'

                sf.write(voice_file_name, wav[int(start * sr): int(end * sr) + int(add_time * sr)], sr)
    
    
def write_json(timeline, piece):
    list_time = []
    for idx, (label, start, end) in enumerate(voise_iterator):
        if min_len_audio < end - start:
            list_time.append({'start_time': start, 'end_time': end})
    data = {'speech_segments': list_time}
    with open(f'{piece}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
            

def from_dir(dir):
    '''
    :param dir: path to directory
    :return: list off files in directory
    '''
    all_files = []
    for _, _, files in os.walk(dir):
        for filename in files:
            all_files.append(dir + '/' + filename)
    return all_files


def cut_wav_on_parts(wav, cuts_on):
    '''
    Cutting a big audio file into parts
    :param wav: "audio.wav" file, which must be cutted into parts
    :param cuts_on: parts duration in seconds (int)
    :return: dict of new files names and info about begin and end of audio
    '''
    audio = AudioSegment.from_wav(wav)
    total_secs = math.ceil(audio.duration_seconds)
    file_names = {}
    iterator = tqdm(range(0, total_secs, args.cuts_on), leave=False)
    for part in iterator:
        from_sec = part
        to_sec = part + cuts_on if part + cuts_on < total_secs else total_secs
        file_name = f'{from_sec}_to_{to_sec}_secs_{os.path.basename(wav)}'
        audio[from_sec * 1000:to_sec * 1000].export(file_name, format='wav')
        iterator.set_description(f'{file_name} is create')
        file_names[file_name] = {'begin': from_sec, 'end': to_sec}

    return file_names, total_secs


def diarize(args):
    if not os.path.exists(f'./{args.dir_save}/'):
        os.mkdir(f'./{args.dir_save}/')

    encoder = VoiceEncoder(args.device, verbose=False)

    def wav_proccecing(args, piece=None, start_from=None, save_file=True, json_file=False):
        '''
        :param args: parameters from console
        :param piece: If not the whole file (only piece) was given (name and path_to_dir)
        :param start_from: for the beauty of new file naming and for correctly display the time interval
        :return: info about postprocessing
        '''

        if isinstance(piece, dict):
            wav, sr_0 = read_file(piece['name'])
        else:
            wav, sr_0 = read_file(piece)

        noise, sr_1 = read_file(args.noise)
        voice, sr_2 = read_file(args.voice)

        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
        noise_similarity = get_similarity(encoder, cont_embeds, noise)
        voice_similarity = get_similarity(encoder, cont_embeds, voice)

        similarity_dict = {'noise': noise_similarity, 'voice': voice_similarity}
        wav_splits_seconds = np.array(
            list(map(lambda x: [x.start / sampling_rate, x.stop / sampling_rate], wav_splits)))

        change_moments, names = get_change_moments(similarity_dict, wav_splits_seconds, args.plot)
        if start_from:
            change_moments = [t for t in change_moments]
    
        diarized_fragments = get_fragment_parts(change_moments, names)
        
        if save_file:
            split_wav(wav, sr_0, diarized_fragments, dir_save=piece['path_to_dir'], add_time=args.add_time,
                  min_len_audio=args.min_len_audio, start_from=start_from)
        if json_file:
            write_json(diarized_fragments, piece)

        return diarized_fragments

    if not os.path.isfile(args.audio):
        files = from_dir(args.audio)
        diarized_fragments = {}
    else:
        files = [args.audio, ]

    file_iterator = tqdm(files)
    for file in file_iterator:
        file_iterator.set_description(f'"{file}" is processing now')
        catalog_per_audio = os.path.splitext(os.path.basename(file))[0]
        if len(files) > 1:
            dir_for_parts = os.path.join(args.dir_save, catalog_per_audio)
            if not os.path.exists(dir_for_parts):
                os.makedirs(dir_for_parts)
        else:
            dir_for_parts = args.dir_save

        if args.cuts_on:
            parts, total_secs = cut_wav_on_parts(file, args.cuts_on)
            piece = {}
            iterator = tqdm(parts.keys(), leave=False)
            for i, part in enumerate(iterator):
                piece['name'] = part
                piece['path_to_dir'] = dir_for_parts

                iterator.set_description(f'{parts[part]["begin"]}-{parts[part]["end"]} sec (total: {total_secs}) processing')

                diarized_fragments[file] = wav_proccecing(args, piece, start_from=i * args.cuts_on) # TYT
                os.remove(part)

        else:
            diarized_fragments = wav_proccecing(args, file)

    return diarized_fragments


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    diarize(args)
