import json
import os.path
from argparse import ArgumentParser

from vad.file_io import load_sound, load_bounds_of_spoken_frames, load_config_data
from vad.vad import detect_spoken_frames_with_webrtc, detect_spoken_frames, calculate_error_of_VAD


def select_sound_files_and_transcriptions(dir_name):
    for cur_name in os.listdir(dir_name):
        if cur_name in {'.', '..'}:
            continue
        full_name = os.path.join(dir_name, cur_name)
        if os.path.isdir(full_name):
            yield from select_sound_files_and_transcriptions(full_name)
        elif cur_name.lower().endswith('.wav'):
            if os.path.exists(full_name[:-4] + '.lab'):
                yield (full_name, full_name[:-4] + '.lab')


def estimate_quality(sound_name, transcription_name, method_of_VAD, params_of_VAD):
    assert (method_of_VAD.lower() == 'webrtc') or (method_of_VAD.lower() == 'adapt'), 'Method of VAD is unknown!'
    sound, fs = load_sound(sound_name)
    transcription = load_bounds_of_spoken_frames(transcription_name)
    if method_of_VAD.lower() == 'webrtc':
        bounds_of_speech = list(detect_spoken_frames_with_webrtc(sound, fs, params_of_VAD))
    else:
        bounds_of_speech = list(detect_spoken_frames(sound, fs, params_of_VAD))
    return calculate_error_of_VAD(len(sound), bounds_of_speech, transcription, fs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_dir', help='Source directory with sound files in WAVE PCM format.',
                        required=True)
    parser.add_argument('-c', '--config', dest='config_json',
                        help='JSON file with configuration parameters of VAD.', required=True)
    args = parser.parse_args()

    source_dir = os.path.normpath(args.source_dir)
    assert os.path.isdir(source_dir), 'Source directory with sound files in WAVE PCM format does not exist!'
    config_name = os.path.normpath(args.config_json)
    assert os.path.isfile(config_name), 'JSON file with configuration parameters of VAD does not exist!'
    method_name, method_params = load_config_data(config_name)

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    nsounds = 0
    for cur_pair in select_sound_files_and_transcriptions(source_dir):
        quality = estimate_quality(cur_pair[0], cur_pair[1], method_name, method_params)
        total_precision += quality[0]
        total_recall += quality[1]
        total_f1 += quality[2]
        nsounds += 1
    if nsounds > 0:
        if nsounds == 1:
            print('There is 1 sound with transcription in a specified directory.')
        else:
            print('There are {0} sounds with transcriptions in a specified directory.'.format(nsounds))
        print('Precision:  {0:>7.2%}'.format(total_precision / nsounds))
        print('Recall:     {0:>7.2%}'.format(total_recall / nsounds))
        print('F1-measure: {0:>7.2%}'.format(total_f1 / nsounds))
    else:
        print('There are no sounds with transcriptions in a specified directory!')