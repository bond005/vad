import math
import struct
from functools import reduce

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import signal, stats
import webrtcvad


FRAME_DURATION = 0.01
EPS = 1e-6
INIT_SILENCE_FRAMES = 30
DEFAULT_PARAMS_OF_WEBRTC_ALG = {'Min_Silence': 0.2, 'Min_Speech': 0.1}
DEFAULT_PARAMS_OF_ADAPT_ALG = {'Energy_PrimThresh': 40.0, 'F_PrimThresh': 185, 'SF_PrimThresh': 5.0,
                               'Min_Silence': 0.2, 'Min_Speech': 0.1}

global_features_cache = dict()


def hertz_to_mel(frequency):
    return 1125.0 * math.log(1.0 + frequency / 700.0)

def mel_to_hertz(frequency):
    return 700.0 * (math.exp(frequency / 1125.0) - 1.0)


def calculate_triangle_filters(frequencies_axis, n_filters = 24):
    mel_width = (hertz_to_mel(frequencies_axis[-1]) - hertz_to_mel(frequencies_axis[0])) / float(n_filters + 1)
    mel_bands = [(ind * mel_width, (ind + 1) * mel_width, (ind + 2) * mel_width) for ind in range(n_filters)]
    triangle_filters = numpy.zeros((n_filters, frequencies_axis.shape[0]))
    new_frequencies_axis = numpy.empty(shape=(n_filters,))
    for filter_ind in range(n_filters):
        start_freq = mel_to_hertz(mel_bands[filter_ind][0])
        middle_freq = mel_to_hertz(mel_bands[filter_ind][1])
        end_freq = mel_to_hertz(mel_bands[filter_ind][2])
        for freq_ind in range(frequencies_axis.shape[0]):
            cur_freq = frequencies_axis[freq_ind]
            if (cur_freq < start_freq) or (cur_freq > end_freq):
                continue
            if cur_freq <= middle_freq:
                value = 2.0 * (cur_freq - start_freq) / ((end_freq - start_freq) * (middle_freq - start_freq))
            else:
                value = 2.0 * (end_freq - cur_freq) / ((end_freq - start_freq) * (end_freq - middle_freq))
            triangle_filters[filter_ind][freq_ind] = value
            new_frequencies_axis[filter_ind] = middle_freq
    return triangle_filters, new_frequencies_axis


def smooth_spectrogram(source_spectrogram, frequencies_axis, n_filters):
    triangle_filters, smoothed_frequencies_axis = calculate_triangle_filters(frequencies_axis, n_filters)
    smoothed_spectrogram = numpy.zeros((source_spectrogram.shape[0], n_filters))
    for time_ind in range(smoothed_spectrogram.shape[0]):
        for filter_ind in range(n_filters):
            for freq_ind in range(frequencies_axis.shape[0]):
                smoothed_spectrogram[time_ind][filter_ind] += source_spectrogram[time_ind][freq_ind] *\
                                                              triangle_filters[filter_ind][freq_ind]
    return smoothed_spectrogram, smoothed_frequencies_axis

def calculate_features_for_VAD(sound_frames, frequencies_axis, spectrogram):
    features = numpy.empty((spectrogram.shape[0], 3))
    # smooted_spectrogram, smoothed_frequencies_axis = smooth_spectrogram(spectrogram, frequencies_axis, 24)
    for time_ind in range(spectrogram.shape[0]):
        mean_spectrum = spectrogram[time_ind].mean()
        if mean_spectrum > 0.0:
            sfm = -10.0 * math.log10(stats.gmean(spectrogram[time_ind]) / mean_spectrum)
        else:
            sfm = 0.0
        # max_freq = smoothed_frequencies_axis[smooted_spectrogram[time_ind].argmax()]
        max_freq = frequencies_axis[spectrogram[time_ind].argmax()]
        features[time_ind][0] = numpy.square(sound_frames[time_ind]).mean()
        features[time_ind][1] = sfm
        features[time_ind][2] = max_freq
    """medfilt_order = 3
    for feature_ind in range(features.shape[0]):
        features[feature_ind] = signal.medfilt(features[feature_ind], medfilt_order)"""
    return features


def show_VAD_features(sound_data, sampling_frequency):
    assert sampling_frequency >= 8000, 'Sampling frequency is inadmissible!'
    n_data = len(sound_data)
    assert (n_data > 0) and ((n_data % 2) == 0), 'Sound data are wrong!'
    frame_size = int(round(FRAME_DURATION * float(sampling_frequency)))
    n_fft_points = 2
    while n_fft_points < frame_size:
        n_fft_points *= 2
    sound_signal = numpy.empty((int(n_data / 2),))
    for ind in range(sound_signal.shape[0]):
        sound_signal[ind] = float(struct.unpack('<h', sound_data[(ind * 2):(ind * 2 + 2)])[0])
    frequencies_axis, time_axis, spectrogram = signal.spectrogram(
        sound_signal, fs=sampling_frequency, window='hamming', nperseg=frame_size, noverlap=0, nfft=n_fft_points,
        scaling='spectrum', mode='psd'
    )
    spectrogram = spectrogram.transpose()
    if spectrogram.shape[0] <= INIT_SILENCE_FRAMES:
        return []
    if (sound_signal.shape[0] % frame_size) == 0:
        sound_frames = numpy.reshape(sound_signal, (spectrogram.shape[0], frame_size))
    else:
        sound_frames = numpy.reshape(sound_signal[0:int(sound_signal.shape[0] / frame_size) * frame_size],
                                     (spectrogram.shape[0], frame_size))
    features = calculate_features_for_VAD(sound_frames, frequencies_axis, spectrogram).transpose()
    time_axis = time_axis.transpose()
    del spectrogram
    del frequencies_axis
    plt.subplot(411)
    plt.plot(time_axis, features[0])
    plt.title('Short-time Energy')
    plt.grid(True)
    plt.subplot(412)
    plt.plot(time_axis, features[1])
    plt.title('Spectral Flatness Measure')
    plt.grid(True)
    plt.subplot(413)
    plt.plot(time_axis, features[2])
    plt.title('Most Dominant Frequency Component')
    plt.grid(True)
    plt.subplot(414)
    x = numpy.repeat(time_axis, 4)
    y = []
    for time_ind in range(time_axis.shape[0]):
        y += [sound_frames[time_ind][0], sound_frames[time_ind].max(), sound_frames[time_ind].min(),
              sound_frames[time_ind][-1]]
    y = numpy.array(y)
    plt.plot(x, y)
    plt.title('Wave File')
    plt.grid(True)
    plt.show()
    del sound_frames
    del time_axis


def smooth_spoken_frames(spoken_frames, min_frames_in_silence, min_frames_in_speech):
    n_frames = len(spoken_frames)
    prev_speech_pos = -1
    for frame_ind in range(n_frames):
        if spoken_frames[frame_ind]:
            if prev_speech_pos >= 0:
                if (prev_speech_pos + 1) < frame_ind:
                    spoken_frames[(prev_speech_pos + 1):frame_ind] = [True] * (frame_ind - prev_speech_pos - 1)
            prev_speech_pos = frame_ind
        else:
            if prev_speech_pos >= 0:
                if (frame_ind - prev_speech_pos) > min_frames_in_silence:
                    prev_speech_pos = -1
    if prev_speech_pos >= 0:
        if (prev_speech_pos + 1) < n_frames:
            spoken_frames[(prev_speech_pos + 1):n_frames] = [True] * (n_frames - prev_speech_pos - 1)
    speech_start = -1
    for frame_ind in range(n_frames):
        if spoken_frames[frame_ind]:
            if speech_start < 0:
                speech_start = frame_ind
        else:
            if speech_start >= 0:
                if (frame_ind - speech_start) >= min_frames_in_speech:
                    yield (speech_start, frame_ind)
                speech_start = -1
    if speech_start >= 0:
        if (n_frames - speech_start) >= min_frames_in_speech:
            yield (speech_start, n_frames)


#@profile
def detect_spoken_frames(sound_data, sampling_frequency, params=DEFAULT_PARAMS_OF_ADAPT_ALG, sound_name=None):
    assert sampling_frequency >= 8000, 'Sampling frequency is inadmissible!'
    n_data = len(sound_data)
    assert (n_data > 0) and ((n_data % 2) == 0), 'Sound data are wrong!'
    frame_size = int(round(FRAME_DURATION * float(sampling_frequency)))
    if (sound_name is None) or (sound_name not in global_features_cache):
        n_fft_points = 2
        while n_fft_points < frame_size:
            n_fft_points *= 2
        sound_signal = numpy.empty((int(n_data / 2),))
        for ind in range(sound_signal.shape[0]):
            sound_signal[ind] = float(struct.unpack('<h', sound_data[(ind * 2):(ind * 2 + 2)])[0])
        frequencies_axis, time_axis, spectrogram = signal.spectrogram(
            sound_signal, fs=sampling_frequency, window='hamming', nperseg=frame_size, noverlap=0, nfft=n_fft_points,
            scaling='spectrum', mode='psd'
        )
        spectrogram = spectrogram.transpose()
        if spectrogram.shape[0] <= INIT_SILENCE_FRAMES:
            return []
        if (sound_signal.shape[0] % frame_size) == 0:
            sound_frames = numpy.reshape(sound_signal, (spectrogram.shape[0], frame_size))
        else:
            sound_frames = numpy.reshape(sound_signal[0:int(sound_signal.shape[0] / frame_size) * frame_size],
                                         (spectrogram.shape[0], frame_size))
        features = calculate_features_for_VAD(sound_frames, frequencies_axis, spectrogram)
        del sound_frames
        del spectrogram
        del frequencies_axis
        del time_axis
        if sound_name is not None:
            global_features_cache[sound_name] = features.copy()
    else:
        features = global_features_cache[sound_name]
    [min_energy, min_sfm, min_freq] = features[0:INIT_SILENCE_FRAMES].min(axis=0).tolist()
    energy_th = params['Energy_PrimThresh'] * math.log(min_energy)
    sfm_th = params['SF_PrimThresh']
    freq_th = params['F_PrimThresh']
    spoken_frames = []
    number_of_silence_frames = 0
    for ind in range(features.shape[0]):
        counter = 0
        if (features[ind][0] - min_energy) >= energy_th:
            counter += 1
        if (features[ind][1] - min_sfm) >= sfm_th:
            counter += 1
        if (features[ind][2] - min_freq) >= freq_th:
            counter += 1
        if counter > 1:
            spoken_frames.append(True)
        else:
            spoken_frames.append(False)
            min_energy = (features[ind][0] + number_of_silence_frames * min_energy) / (number_of_silence_frames + 1)
            energy_th = params['Energy_PrimThresh'] * math.log(min_energy)
            number_of_silence_frames += 1
    del features
    min_frames_in_silence = int(round(params['Min_Silence'] * float(sampling_frequency) / frame_size))
    if min_frames_in_silence < 0:
        min_frames_in_silence = 0
    min_frames_in_speech = int(round(params['Min_Speech'] * float(sampling_frequency) / frame_size))
    if min_frames_in_speech < 0:
        min_frames_in_speech = 0
    sound_duration = (n_data - 2.0) / (2.0 * float(sampling_frequency))
    for cur_speech_frame in smooth_spoken_frames(spoken_frames, min_frames_in_silence, min_frames_in_speech):
        init_time = cur_speech_frame[0] * FRAME_DURATION
        fin_time = cur_speech_frame[1] * FRAME_DURATION
        if fin_time > sound_duration:
            fin_time = sound_duration
        yield (init_time, fin_time)
    del spoken_frames


#@profile
def detect_spoken_frames_with_webrtc(sound_data, sampling_frequency, params=DEFAULT_PARAMS_OF_WEBRTC_ALG):
    assert sampling_frequency in (8000, 16000, 32000), 'Sampling frequency is inadmissible!'
    n_data = len(sound_data)
    assert (n_data > 0) and ((n_data % 2) == 0), 'Sound data are wrong!'
    frame_size = int(round(FRAME_DURATION * float(sampling_frequency)))
    sound_duration = (n_data - 2.0) / (2.0 * float(sampling_frequency))
    n_frames = int(round(n_data / (2.0 * float(frame_size))))
    spoken_frames = [False] * n_frames
    buffer_start = 0
    vad = webrtcvad.Vad(mode=3)
    for frame_ind in range(n_frames):
        if (buffer_start + frame_size * 2) <= n_data:
            if vad.is_speech(sound_data[buffer_start:(buffer_start + frame_size * 2)],
                             sample_rate=sampling_frequency):
                spoken_frames[frame_ind] = True
        buffer_start += (frame_size * 2)
    del vad
    min_frames_in_silence = int(round(params['Min_Silence'] * float(sampling_frequency) / frame_size))
    if min_frames_in_silence < 0:
        min_frames_in_silence = 0
    min_frames_in_speech = int(round(params['Min_Speech'] * float(sampling_frequency) / frame_size))
    if min_frames_in_speech < 0:
        min_frames_in_speech = 0
    for cur_speech_frame in smooth_spoken_frames(spoken_frames, min_frames_in_silence, min_frames_in_speech):
        init_time = cur_speech_frame[0] * FRAME_DURATION
        fin_time = cur_speech_frame[1] * FRAME_DURATION
        if fin_time > sound_duration:
            fin_time = sound_duration
        yield (init_time, fin_time)
    del spoken_frames


def calculate_energy(sound_data):
    n_data = len(sound_data)
    assert (n_data > 0) and ((n_data % 2) == 0), 'Sound data are wrong!'
    n_samples = int(n_data / 2)
    total_energy = reduce(
        lambda energy, cur_sample: energy + cur_sample * cur_sample,
        map(
            lambda sample_ind: float(struct.unpack('<h', sound_data[(sample_ind * 2):(sample_ind * 2 + 2)])[0]),
            range(n_samples)
        ),
        0.0
    )
    return total_energy, n_samples


def calculate_SNR(sound_data, sampling_frequency, bounds_of_spoken_frames):
    n_data = len(sound_data)
    assert (n_data > 0) and ((n_data % 2) == 0), 'Sound data are wrong!'
    if len(bounds_of_spoken_frames) == 0:
        return None
    n_samples = int(n_data / 2)
    speech_energy = 0.0
    number_of_speech_samples = 0
    noise_energy = 0.0
    number_of_noise_samples = 0
    prev_speech_end = 0
    for bounds_of_cur_frame in bounds_of_spoken_frames:
        cur_speech_start = int(round(bounds_of_cur_frame[0] * sampling_frequency))
        cur_speech_end = int(round(bounds_of_cur_frame[1] * sampling_frequency))
        if cur_speech_start >= n_samples:
            break
        if cur_speech_end > n_samples:
            cur_speech_end = n_samples
        if cur_speech_start > prev_speech_end:
            frame_energy, samples_in_frame = calculate_energy(sound_data[(prev_speech_end * 2):(cur_speech_start * 2)])
            noise_energy += frame_energy
            number_of_noise_samples += samples_in_frame
        if cur_speech_end > cur_speech_start:
            frame_energy, samples_in_frame = calculate_energy(sound_data[(cur_speech_start * 2):(cur_speech_end * 2)])
            speech_energy += frame_energy
            number_of_speech_samples += samples_in_frame
        prev_speech_end = cur_speech_end
    if n_samples > prev_speech_end:
        frame_energy, samples_in_frame = calculate_energy(sound_data[(prev_speech_end * 2):(n_samples * 2)])
        noise_energy += frame_energy
        number_of_noise_samples += samples_in_frame
    if (number_of_noise_samples == 0) or (number_of_speech_samples == 0):
        return None
    speech_energy = speech_energy / float(number_of_speech_samples) + EPS
    noise_energy = noise_energy / float(number_of_noise_samples) + EPS
    return 20.0 * math.log10(speech_energy / noise_energy)


def calculate_error_of_VAD(nbytes_in_sound, recognized_bounds, ideal_bounds, sampling_frequency):
    assert (nbytes_in_sound > 0) and ((nbytes_in_sound % 2) == 0), 'Number of bytes in sound is incorrect!'
    assert sampling_frequency >= 8000, 'Sampling frequency is inadmissible!'
    frame_size = int(round(FRAME_DURATION * float(sampling_frequency)))
    n_frames = int(round(nbytes_in_sound / (2.0 * float(frame_size))))
    results_of_recognition = [False] * n_frames
    for bounds_of_cur_frame in recognized_bounds:
        start_sample_ind = int(round(bounds_of_cur_frame[0] * sampling_frequency))
        end_sample_ind = int(round(bounds_of_cur_frame[1] * sampling_frequency)) - 1
        start_frame_ind = int(start_sample_ind / frame_size)
        end_frame_ind = int(end_sample_ind / frame_size)
        for frame_ind in range(end_frame_ind - start_frame_ind + 1):
            results_of_recognition[start_frame_ind + frame_ind] = True
    ideal_results = [False] * n_frames
    for bounds_of_cur_frame in ideal_bounds:
        start_sample_ind = int(round(bounds_of_cur_frame[0] * sampling_frequency))
        end_sample_ind = int(round(bounds_of_cur_frame[1] * sampling_frequency)) - 1
        start_frame_ind = int(start_sample_ind / frame_size)
        end_frame_ind = int(end_sample_ind / frame_size)
        for frame_ind in range(end_frame_ind - start_frame_ind + 1):
            ideal_results[start_frame_ind + frame_ind] = True
    res = (
        precision_score(ideal_results, results_of_recognition),
        recall_score(ideal_results, results_of_recognition),
        f1_score(ideal_results, results_of_recognition)
    )
    del ideal_results
    del results_of_recognition
    return res


def clear_cache():
    global_features_cache.clear()
