import copy
import itertools
import json
import os.path
from argparse import ArgumentParser

#from memory_profiler import profile

from vad.file_io import load_timit_sound, load_timit_bounds_of_spoken_frames
from vad.vad import detect_spoken_frames_with_webrtc, detect_spoken_frames, calculate_error_of_VAD
from vad.vad import DEFAULT_PARAMS_OF_ADAPT_ALG, DEFAULT_PARAMS_OF_WEBRTC_ALG


def generate_variants_of_param(param_description, param_type):
    list_of_parameter_values = []
    error_msg = '"{0}": this is incorrect description of parameter variants!'.format(param_description)
    parts_of_description = param_description.split(':')
    assert len(parts_of_description) == 3, error_msg
    ok = True
    try:
        min_value = param_type(parts_of_description[0].strip())
        step = param_type(parts_of_description[1].strip())
        max_value = param_type(parts_of_description[2].strip())
        if (min_value >= max_value) or (step <= 0):
            ok = False
        else:
            if step >= (max_value - min_value):
                nrepeats = 1
            else:
                nrepeats = int(round((max_value - min_value) / step))
            step = (max_value - min_value) / float(nrepeats)
            nrepeats += 1
            list_of_parameter_values = [min_value + step * n for n in range(nrepeats)]
    except:
        ok = False
    assert ok, error_msg
    return tuple(list_of_parameter_values)


def generate_all_combinations_of_params(variants_of_params, names_of_params):
    values_of_params = [variants_of_params[cur_name] for cur_name in names_of_params]
    for cur_combination in itertools.product(*values_of_params):
        yield cur_combination


def select_sound_files_and_transcriptions(dir_name):
    for cur_name in os.listdir(dir_name):
        if cur_name in {'.', '..'}:
            continue
        full_name = os.path.join(dir_name, cur_name)
        if os.path.isdir(full_name):
            yield from select_sound_files_and_transcriptions(full_name)
        elif cur_name.lower().endswith('.wav'):
            if os.path.exists(full_name[:-4] + '.phn'):
                sound, fs = load_timit_sound(full_name)
                transcription = load_timit_bounds_of_spoken_frames(full_name[:-4] + '.phn')
                yield (sound, fs, transcription, full_name)
                del sound
                del transcription


#@profile
def estimate_quality(sound_data, sampling_frequency, timit_transcription, method_of_VAD, alg_params, sound_name=None):
    assert (method_of_VAD.lower() == 'webrtc') or (method_of_VAD.lower() == 'adapt'), 'Method of VAD is unknown!'
    if method_of_VAD.lower() == 'webrtc':
        speech_bounds = detect_spoken_frames_with_webrtc(sound_data, sampling_frequency, alg_params)
    else:
        speech_bounds = detect_spoken_frames(sound_data, sampling_frequency, alg_params, sound_name)
    quality = calculate_error_of_VAD(len(sound_data), speech_bounds, timit_transcription, sampling_frequency)
    return quality


if __name__ == '__main__':
    format_msg = 'in format "<initial_value>:<step><final_value>".'
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_dir', help='Source directory with sound files in WAVE PCM format.',
                        required=True)
    parser.add_argument('-m', '--method', dest='method_name', help='Name of method for voice activity detection.',
                        required=True)
    parser.add_argument('-d', '--dst', dest='destination_json',
                        help='Destination JSON file with optimal parameters of VAD.', required=True)
    parser.add_argument('--min_silence', dest='min_silence',
                        help='Variants of a minimal duration of silence (in seconds) ' + format_msg,
                        required=False)
    parser.add_argument('--min_speech', dest='min_speech',
                        help='Variants of a minimal duration of spoken frame (in seconds) ' + format_msg,
                        required=False)
    parser.add_argument('--energy', dest='energy', help='Variants of an energy primary threshold ' + format_msg,
                        required=False)
    parser.add_argument('--sfm', dest='sfm', help='Variants of a SFM primary threshold ' + format_msg, required=False)
    parser.add_argument('--frequency', dest='frequency', help='Variants of a frequency primary threshold ' + format_msg,
                        required=False)
    args = parser.parse_args()

    source_dir = os.path.normpath(args.source_dir)
    assert os.path.isdir(source_dir), 'Source directory with sound files in WAVE PCM format does not exist!'
    destination_json_name = os.path.normpath(args.destination_json)
    assert os.path.isdir(os.path.dirname(destination_json_name)), ''
    method_name = args.method_name.lower()
    assert method_name in {'webrtc', 'adapt'}, 'Method of VAD is unknown!'
    variants_of_params = dict()
    if method_name == 'webrtc':
        for cur_param in DEFAULT_PARAMS_OF_WEBRTC_ALG:
            variants_of_params[cur_param] = (DEFAULT_PARAMS_OF_WEBRTC_ALG[cur_param],)
    else:
        for cur_param in DEFAULT_PARAMS_OF_ADAPT_ALG:
            variants_of_params[cur_param] = (DEFAULT_PARAMS_OF_ADAPT_ALG[cur_param],)
    if args.min_silence is not None:
        variants_of_params['Min_Silence'] = generate_variants_of_param(args.min_silence, float)
    if args.min_speech is not None:
        variants_of_params['Min_Speech'] = generate_variants_of_param(args.min_speech, float)
    if args.energy is not None:
        variants_of_params['Energy_PrimThresh'] = generate_variants_of_param(args.energy, float)
    if args.sfm is not None:
        variants_of_params['SF_PrimThresh'] = generate_variants_of_param(args.sfm, float)
    if args.frequency is not None:
        variants_of_params['F_PrimThresh'] = generate_variants_of_param(args.frequency, int)

    sounds_and_transcriptions = list(select_sound_files_and_transcriptions(source_dir))
    nsounds = len(sounds_and_transcriptions)
    assert nsounds > 0, 'There are no sounds with transcriptions in a specified directory!'
    if nsounds == 1:
        print('There is 1 sound with transcription in a specified directory.')
    else:
        print('There are {0} sounds with transcriptions in a specified directory.'.format(nsounds))
    best_total_f1 = None
    best_combination_of_parameters = dict()
    names_of_params = sorted(list(variants_of_params.keys()))
    width_of_param_name = len(names_of_params[0])
    for param_name in names_of_params[1:]:
        if len(param_name) > width_of_param_name:
            width_of_param_name = len(param_name)
    for cur_combination_of_values in generate_all_combinations_of_params(variants_of_params, names_of_params):
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        combination_dictionary = dict(zip(names_of_params, cur_combination_of_values))
        for cur_pair in sounds_and_transcriptions:
            quality = estimate_quality(cur_pair[0], cur_pair[1], cur_pair[2], method_name, combination_dictionary,
                                       cur_pair[3])
            total_precision += quality[0]
            total_recall += quality[1]
            total_f1 += quality[2]
        print('')
        print('Parameters of algorithm:')
        for ind in range(len(names_of_params)):
            print('  {0:{1}}: {2}'.format(names_of_params[ind], width_of_param_name, cur_combination_of_values[ind]))
        total_precision /= nsounds
        total_recall /= nsounds
        total_f1 /= nsounds
        print('Precision:  {0:>7.2%}'.format(total_precision))
        print('Recall:     {0:>7.2%}'.format(total_recall))
        print('F1-measure: {0:>7.2%}'.format(total_f1))
        if best_total_f1 is None:
            best_total_f1 = total_f1
            best_combination_of_parameters = copy.copy(combination_dictionary)
        else:
            if total_f1 > best_total_f1:
                best_total_f1 = total_f1
                best_combination_of_parameters = copy.copy(combination_dictionary)
        del cur_combination_of_values
        del combination_dictionary
    assert best_total_f1 is not None, 'Best combination of VAD parameters cannot be calculated!'
    best_combination_of_parameters['f1'] = best_total_f1
    best_combination_of_parameters['method'] = method_name
    with open(destination_json_name, 'w') as json_fp:
        json.dump(best_combination_of_parameters, json_fp, indent=4, sort_keys=True)
