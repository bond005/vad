import codecs
import os.path
from argparse import ArgumentParser

from vad.file_io import load_sound, load_config_data
from vad.vad import detect_spoken_frames_with_webrtc, detect_spoken_frames, calculate_SNR


def select_sound_files_and_their_SNR(dir_name, method_name, params_of_VAD):
    for cur_name in os.listdir(dir_name):
        if cur_name in {'.', '..'}:
            continue
        full_name = os.path.join(dir_name, cur_name)
        if os.path.isdir(full_name):
            yield from select_sound_files_and_their_SNR(full_name, method_name, params_of_VAD)
        elif cur_name.lower().endswith('.wav'):
            sound, fs = load_sound(full_name)
            if method_name == 'adapt':
                bounds_of_speech = list(detect_spoken_frames(sound, fs, params_of_VAD))
            else:
                bounds_of_speech = list(detect_spoken_frames_with_webrtc(sound, fs, params_of_VAD))
            snr = calculate_SNR(sound, fs, bounds_of_speech)
            if snr is not None:
                yield (full_name, int(round(snr)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_dir', help='Source directory with sound files in WAVE PCM format.',
                        required=True)
    parser.add_argument('-d', '--dst', dest='destination_name',
                        help='Destination text file with full names of sound files which are sorted by decrease of '
                             'signal-to-noise ratio.',
                        required=True)
    parser.add_argument('-c', '--config', dest='config_json',
                        help='JSON file with configuration parameters of VAD.', required=True)
    args = parser.parse_args()

    source_dir = os.path.normpath(args.source_dir)
    assert os.path.isdir(source_dir), 'Source directory with sound files in WAVE PCM format does not exist!'
    report_name = os.path.normpath(args.destination_name)
    assert (len(os.path.dirname(report_name)) == 0) or (os.path.isdir(os.path.dirname(report_name))), \
        'Directory into which the destination text file does not exist!'
    config_name = os.path.normpath(args.config_json)
    assert os.path.isfile(config_name), 'JSON file with configuration parameters of VAD does not exist!'
    method_name, method_params = load_config_data(config_name)

    sounds_and_SNR = sorted(
        list(select_sound_files_and_their_SNR(source_dir, method_name, method_params)),
        key=lambda a: (-a[1], a[0])
    )
    max_length_of_name = 0
    max_length_of_snr = 0
    for cur_sound_and_SNR in sounds_and_SNR:
        name_length = len(cur_sound_and_SNR[0])
        if name_length > max_length_of_name:
            max_length_of_name = name_length
        snr_length = len(str(cur_sound_and_SNR[1]))
        if snr_length > max_length_of_name:
            max_length_of_snr = snr_length
    with codecs.open(report_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        for cur_sound_and_SNR in sounds_and_SNR:
            fp.write('{0:<{1}} {2:>{3}}\n'.format(cur_sound_and_SNR[0], max_length_of_name, cur_sound_and_SNR[1],
                                                  max_length_of_snr))
