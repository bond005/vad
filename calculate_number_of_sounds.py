import os.path
from argparse import ArgumentParser


def calculate_sound_files_in_directory(dir_name):
    number_of_sound_files = 0
    for cur_name in os.listdir(dir_name):
        if cur_name in {'.', '..'}:
            continue
        full_name = os.path.join(dir_name, cur_name)
        if os.path.isdir(full_name):
            number_of_sound_files += calculate_sound_files_in_directory(full_name)
        if cur_name.lower().endswith('.wav'):
            number_of_sound_files += 1
    return number_of_sound_files


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_dir', help='Source directory with sound files in WAVE PCM format.',
                        required=True)
    args = parser.parse_args()

    source_dir = os.path.normpath(args.source_dir)
    assert os.path.isdir(source_dir), 'Source directory with sound files in WAVE PCM format does not exist!'

    n = calculate_sound_files_in_directory(source_dir)
    print('Number of WAV PCM files in a specified directory and all its subdirectories is {0}.'.format(n))
