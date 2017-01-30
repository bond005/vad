from functools import reduce
import math
import struct
import unittest

import numpy
from scipy import signal

from vad import vad


class TestVAD(unittest.TestCase):
    def test_calculate_energy_positive_1(self):
        source_array = (89, -89, 89, -89, 89, -89)
        source_data = reduce(
            lambda a, b: a + struct.pack('>h', b), source_array[1:], struct.pack('>h', source_array[0])
        )
        target_energy = reduce(lambda a, b: a + b * b, source_array, 0) / float(len(source_array))
        calculated_energy = vad.calculate_energy(source_data)
        self.assertIsInstance(calculated_energy, tuple)
        self.assertEqual(2, len(calculated_energy))
        self.assertAlmostEqual(target_energy, calculated_energy[0] / calculated_energy[1])

    def test_calculate_energy_negative_1(self):
        source_array = (89, -89, 89, -89, 89, -89)
        source_data = reduce(
            lambda a, b: a + struct.pack('>h', b), source_array[1:], struct.pack('>h', source_array[0])
        )
        with self.assertRaisesRegex(AssertionError, 'Sound data are wrong!'):
            vad.calculate_energy(source_data[0:-1])

    def test_calculate_energy_negative_2(self):
        with self.assertRaisesRegex(AssertionError, 'Sound data are wrong!'):
            vad.calculate_energy([])

    def test_calculate_SNR_positive_1(self):
        source_array = [89, -89] * 6000 + [502, -502] * 8000 + [89, -89] * 7000
        source_data = reduce(
            lambda a, b: a + struct.pack('>h', b), source_array[1:], struct.pack('>h', source_array[0])
        )
        sampling_frequency = 8000
        bounds_of_speech = [(2.0 * 6000.0 / sampling_frequency, 2.0 * (6000.0 + 8000.0) / sampling_frequency)]
        silence_energy = reduce(
            lambda a, b: a + b * b,
            source_array[0:(2 * 6000)] + source_array[(2 * (6000 + 8000)):],
            vad.EPS
        ) / (2.0 * (6000.0 + 7000.0))
        speech_energy = reduce(
            lambda a, b: a + b * b,
            source_array[(2 * 6000):(2 * (6000 + 8000))],
            vad.EPS
        ) / (2.0 * 8000.0)
        target_snr = 20.0 * math.log10(speech_energy / silence_energy)
        self.assertAlmostEqual(target_snr, vad.calculate_SNR(source_data, sampling_frequency, bounds_of_speech))

    def test_calculate_SNR_negative_1(self):
        source_array = [89, -89] * 6000 + [502, -502] * 8000 + [89, -89] * 7000
        source_data = reduce(
            lambda a, b: a + struct.pack('>h', b), source_array[1:], struct.pack('>h', source_array[0])
        )
        sampling_frequency = 8000
        bounds_of_speech = [(2.0 * 6000.0 / sampling_frequency, 2.0 * (6000.0 + 8000.0) / sampling_frequency)]
        with self.assertRaisesRegex(AssertionError, 'Sound data are wrong!'):
            vad.calculate_SNR(source_data[0:-1], sampling_frequency, bounds_of_speech)

    def test_calculate_SNR_negative_2(self):
        sampling_frequency = 8000
        bounds_of_speech = [(2.0 * 6000.0 / sampling_frequency, 2.0 * (6000.0 + 8000.0) / sampling_frequency)]
        with self.assertRaisesRegex(AssertionError, 'Sound data are wrong!'):
            vad.calculate_SNR([], sampling_frequency, bounds_of_speech)

    def test_calculate_SNR_negative_3(self):
        source_array = [89, -89] * 6000 + [502, -502] * 8000 + [89, -89] * 7000
        source_data = reduce(
            lambda a, b: a + struct.pack('>h', b), source_array[1:], struct.pack('>h', source_array[0])
        )
        sampling_frequency = 8000
        bounds_of_speech = [(106000.0 / sampling_frequency, (106000.0 + 8000.0) / sampling_frequency)]
        self.assertIsNone(vad.calculate_SNR(source_data, sampling_frequency, bounds_of_speech))

    def test_calculate_SNR_negative_4(self):
        source_array = [89, -89] * 6000 + [502, -502] * 8000 + [89, -89] * 7000
        source_data = reduce(
            lambda a, b: a + struct.pack('>h', b), source_array[1:], struct.pack('>h', source_array[0])
        )
        sampling_frequency = 8000
        self.assertIsNone(vad.calculate_SNR(source_data, sampling_frequency, []))

    def test_calculate_SNR_negative_5(self):
        source_array = [89, -89] * 6000 + [502, -502] * 8000 + [89, -89] * 7000
        source_data = reduce(
            lambda a, b: a + struct.pack('>h', b), source_array[1:], struct.pack('>h', source_array[0])
        )
        sampling_frequency = 8000
        bounds_of_speech = [(0.0 / sampling_frequency, (2 * 6000.0 + 2 * 8000.0 + 2 * 7000.0) / sampling_frequency)]
        self.assertIsNone(vad.calculate_SNR(source_data, sampling_frequency, bounds_of_speech))

    def test_smooth_spoken_frames(self):
        source_spoken_frames = [False] * 100 + [True] * 10 + [False] * 120 + [True] * 60 + [False] * 5 + [True] * 35 +\
                               [False] * 180
        min_frames_in_silence = 30
        min_frames_in_speech = 20
        target_bounds = [(230, 330)]
        calculated_bounds = list(vad.smooth_spoken_frames(source_spoken_frames, min_frames_in_silence,
                                                          min_frames_in_speech))
        self.assertEqual(len(target_bounds), len(calculated_bounds))
        for speech_frame_ind in range(len(target_bounds)):
            self.assertIsInstance(calculated_bounds[speech_frame_ind], tuple)
            self.assertEqual(2, len(calculated_bounds[speech_frame_ind]))
            self.assertEqual(target_bounds[speech_frame_ind], calculated_bounds[speech_frame_ind])

    def test_calculate_features_for_VAD(self):
        sampling_frequency = 8000
        frequency1 = 400.0
        period1 = (1.0 / frequency1) * sampling_frequency
        frequency2 = 800.0
        period2 = (1.0 / frequency2) * sampling_frequency
        frame_size = 80
        source_signal = numpy.array([math.sin((ind / period1) * 2.0 * math.pi) for ind in range(16000)])
        for ind in range(4000):
            source_signal[4000 + ind] += 3.0 * math.sin((ind / period2) * 2.0 * math.pi)
        for ind in range(16000):
            if (ind % 2) == 0:
                source_signal[ind] += vad.EPS
            else:
                source_signal[ind] -= vad.EPS
        frequencies_axis, time_axis, spectrogram = signal.spectrogram(
            source_signal, fs=sampling_frequency, window='hamming', nperseg=frame_size, noverlap=0, nfft=128,
            scaling='spectrum', mode='psd'
        )
        spectrogram = spectrogram.transpose()
        frequency1_ind = 0
        frequency2_ind = 0
        for ind in range(frequencies_axis.shape[0]):
            if abs(frequency1 - frequencies_axis[ind]) < abs(frequency1 - frequencies_axis[frequency1_ind]):
                frequency1_ind = ind
            if abs(frequency2 - frequencies_axis[ind]) < abs(frequency2 - frequencies_axis[frequency2_ind]):
                frequency2_ind = ind
        sound_frames = numpy.empty((spectrogram.shape[0], frame_size))
        for ind in range(sound_frames.shape[0]):
            sound_frames[ind] = source_signal[(ind * frame_size) : ((ind + 1) * frame_size)]
        start_frame_ind = int(4000 / frame_size)
        end_frame_ind = int(8000 / frame_size)
        calculated_features = vad.calculate_features_for_VAD(sound_frames, frequencies_axis, spectrogram)
        self.assertIsInstance(calculated_features, numpy.ndarray)
        self.assertEqual((spectrogram.shape[0], 3), calculated_features.shape)
        for ind in range(start_frame_ind):
            self.assertAlmostEqual(frequencies_axis[frequency1_ind], calculated_features[ind][2])
            self.assertTrue(calculated_features[ind][0] > 0.0)
            self.assertTrue(calculated_features[ind][1] > 0.0)
            self.assertNotAlmostEqual(calculated_features[ind][0], calculated_features[ind][1])
            self.assertAlmostEqual(numpy.square(sound_frames[ind]).mean(), calculated_features[ind][0])
        for ind in range(spectrogram.shape[0] - end_frame_ind):
            self.assertAlmostEqual(frequencies_axis[frequency1_ind], calculated_features[ind + end_frame_ind][2])
            self.assertTrue(calculated_features[ind + end_frame_ind][0] > 0.0)
            self.assertTrue(calculated_features[ind + end_frame_ind][1] > 0.0)
            self.assertNotAlmostEqual(calculated_features[ind + end_frame_ind][0],
                                      calculated_features[ind + end_frame_ind][1])
            self.assertAlmostEqual(numpy.square(sound_frames[ind + end_frame_ind]).mean(),
                                   calculated_features[ind + end_frame_ind][0])
        for ind in range(end_frame_ind - start_frame_ind):
            self.assertAlmostEqual(frequencies_axis[frequency2_ind], calculated_features[ind + start_frame_ind][2])
            self.assertTrue(calculated_features[ind + start_frame_ind][0] > 0.0)
            self.assertTrue(calculated_features[ind + start_frame_ind][1] > 0.0)
            self.assertNotAlmostEqual(calculated_features[ind + start_frame_ind][0],
                                      calculated_features[ind + start_frame_ind][1])
            self.assertAlmostEqual(numpy.square(sound_frames[ind + start_frame_ind]).mean(),
                                   calculated_features[ind + start_frame_ind][0])
            for ind2 in range(start_frame_ind):
                self.assertGreater(calculated_features[ind + start_frame_ind][0], calculated_features[ind2][0])
                self.assertLess(calculated_features[ind + start_frame_ind][1], calculated_features[ind2][1])
            for ind2 in range(spectrogram.shape[0] - end_frame_ind):
                self.assertGreater(calculated_features[ind + start_frame_ind][0],
                                   calculated_features[ind2 + end_frame_ind][0])
                self.assertLess(calculated_features[ind + start_frame_ind][1],
                                calculated_features[ind2 + end_frame_ind][1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
