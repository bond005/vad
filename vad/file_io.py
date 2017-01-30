import json
import wave


def load_sound(file_name):
    fp = wave.open(file_name, 'rb')
    try:
        assert fp.getnchannels() == 1, '{0}: sound format is incorrect! Sound must be mono.'.format(file_name)
        assert fp.getsampwidth() == 2, '{0}: sound format is incorrect! ' \
                                       'Sample width of sound must be 2 bytes.'.format(file_name)
        assert fp.getframerate() in (8000, 16000, 32000), '{0}: sound format is incorrect! ' \
                                                          'Sampling frequency must be 8000 Hz, 16000 Hz or 32000 Hz.'
        sampling_frequency = fp.getframerate()
        sound_data = fp.readframes(fp.getnframes())
    finally:
        fp.close()
        del fp
    return sound_data, sampling_frequency


def load_timit_sound(file_name):
    timit_sampling_frequency = 16000
    fp = open(file_name, 'rb')
    try:
        sound_data = fp.read()
        n_data = len(sound_data)
        assert (n_data > 1024) and ((n_data % 2) == 0), '{0}: sound format is incorrect!'.format(file_name)
    finally:
        fp.close()
        del fp
    return sound_data[1024:], timit_sampling_frequency


def save_bounds_of_spoken_frames(file_name, bounds_of_spoken_frames):
    fp = open(file_name, 'w')
    try:
        for bounds_of_cur_frame in bounds_of_spoken_frames:
            fp.write('{0:.7f} {1:.7f} speech\n'.format(bounds_of_cur_frame[0], bounds_of_cur_frame[1]))
        fp.write('\n')
    finally:
        fp.close()
        del fp


def load_bounds_of_spoken_frames(file_name):
    bounds_of_spoken_frames = list()
    fp = open(file_name, 'r')
    try:
        cur_line = fp.readline()
        line_index = 1
        while len(cur_line) > 0:
            prepared_line = cur_line.strip()
            if len(prepared_line) > 0:
                parts_of_line = prepared_line.split()
                error_msg = 'Line {0} in file "{1}": line is incorrect!'.format(line_index, file_name)
                assert len(parts_of_line) == 3, error_msg
                assert parts_of_line[2].lower() == 'speech', error_msg
                ok = True
                try:
                    start_time = float(parts_of_line[0])
                    end_time = float(parts_of_line[1])
                    if end_time <= start_time:
                        ok = False
                    else:
                        bounds_of_spoken_frames.append((start_time, end_time))
                except:
                    ok = False
                assert ok, error_msg
            cur_line = fp.readline()
            line_index += 1
    finally:
        fp.close()
        del fp
    return bounds_of_spoken_frames


def load_timit_bounds_of_spoken_frames(file_name):
    timit_sampling_frequency = 16000.0
    variants_of_silence = {'pau', 'epi', 'h#'}
    speech_start = None
    speech_end = None
    frame_start = None
    frame_end = None
    bounds_of_spoken_frames = list()
    fp = open(file_name, 'r')
    try:
        cur_line = fp.readline()
        line_index = 1
        while len(cur_line) > 0:
            prepared_line = cur_line.strip()
            if len(prepared_line) > 0:
                parts_of_line = prepared_line.split()
                error_msg = 'Line {0} in file "{1}": line is incorrect!'.format(line_index, file_name)
                assert len(parts_of_line) == 3, error_msg
                ok = True
                try:
                    frame_start = int(parts_of_line[0])
                    frame_end = int(parts_of_line[1])
                    if (frame_start < 0) or (frame_end <= frame_start):
                        ok = False
                except:
                    ok = False
                assert ok, error_msg
                if speech_start is None:
                    if parts_of_line[2].lower() not in variants_of_silence:
                        speech_start = frame_start
                        speech_end = frame_end
                else:
                    if parts_of_line[2].lower() in variants_of_silence:
                        bounds_of_spoken_frames.append(
                            (speech_start / timit_sampling_frequency, speech_end / timit_sampling_frequency)
                        )
                        speech_start = None
                        speech_end = None
                    else:
                        speech_end = frame_end
            cur_line = fp.readline()
            line_index += 1
    finally:
        fp.close()
        del fp
    if (speech_start is not None) and (speech_end is not None):
        bounds_of_spoken_frames.append(
            (speech_start / timit_sampling_frequency, speech_end / timit_sampling_frequency)
        )
    return bounds_of_spoken_frames

def load_config_data(config_file_name):
    with open(config_file_name, 'r') as fp:
        config_data = json.load(fp)
    method_name = config_data['method']
    assert method_name in {'webrtc', 'adapt'}, 'Method of VAD is unknown!'
    min_silence = float(config_data['Min_Silence'])
    min_speech = float(config_data['Min_Speech'])
    assert min_silence > 0.0, 'The Min_Silence parameter of configuration is incorrect!'
    assert min_speech > 0.0, 'The Min_Speech parameter configuration is incorrect!'
    params_dictionary = {'Min_Silence': min_silence, 'Min_Speech': min_speech}
    if method_name == 'adapt':
        energy_th = float(config_data['Energy_PrimThresh'])
        sfm_th = float(config_data['SF_PrimThresh'])
        dominant_freq_th = float(config_data['F_PrimThresh'])
        assert energy_th > 0.0, 'The Energy_PrimThresh parameter of configuration is incorrect!'
        assert sfm_th > 0.0, 'The SF_PrimThresh parameter of configuration is incorrect!'
        assert dominant_freq_th > 0.0, 'The F_PrimThresh parameter of configuration is incorrect!'
        params_dictionary['Energy_PrimThresh'] = energy_th
        params_dictionary['SF_PrimThresh'] = sfm_th
        params_dictionary['F_PrimThresh'] = dominant_freq_th
    return method_name, params_dictionary
