import os.path
from argparse import ArgumentParser

from vad.file_io import load_sound, save_bounds_of_spoken_frames, load_config_data
from vad.vad import detect_spoken_frames_with_webrtc, detect_spoken_frames


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_name', help='Source sound file in WAVE PCM format.', required=True)
    parser.add_argument('-d', '--dst', dest='destination_name',
                        help='Destination text file with time bounds of all spoken frames.', required=True)
    parser.add_argument('-c', '--config', dest='config_json',
                        help='JSON file with configuration parameters of VAD.', required=True)
    args = parser.parse_args()

    sound_name = os.path.normpath(args.source_name)
    assert os.path.exists(sound_name), 'Source sound file in WAVE PCM format does not exist!'
    config_name = os.path.normpath(args.config_json)
    assert os.path.isfile(config_name), 'JSON file with configuration parameters of VAD does not exist!'

    method_name, method_params = load_config_data(config_name)
    transcription_name = os.path.normpath(args.destination_name)
    transcription_dir = os.path.dirname(transcription_name)
    assert (len(transcription_dir) == 0) or (os.path.isdir(transcription_dir)),\
        'Directory into which the destination text file with time bounds of all spoken frames will be written ' \
        'does not exist!'

    sound, fs = load_sound(sound_name)
    if method_name == 'webrtc':
        bounds_of_speech = list(detect_spoken_frames_with_webrtc(sound, fs, method_params))
    else:
        bounds_of_speech = list(detect_spoken_frames(sound, fs, method_params))
    save_bounds_of_spoken_frames(transcription_name, bounds_of_speech)
