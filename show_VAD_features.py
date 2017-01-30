import os.path
from argparse import ArgumentParser

from vad.file_io import load_sound, load_timit_sound
from vad.vad import show_VAD_features


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_file', help='Source sound file.', required=True)
    parser.add_argument('-f', '--fmt', dest='format', help='Sound format (wav of timit).', required=True)
    args = parser.parse_args()

    source_sound_name = os.path.normpath(args.source_file)
    assert os.path.exists(source_sound_name), 'Source sound file in WAVE PCM format does not exist!'
    format_name = args.format.strip().lower()
    assert format_name in ['wav', 'timit'], 'Sound format is unknown!'

    if format_name == 'wav':
        sound, fs = load_sound(source_sound_name)
    else:
        sound, fs = load_timit_sound(source_sound_name)
    show_VAD_features(sound, fs)
