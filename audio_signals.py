import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy import stats as sp_stats


class TSignal:
    def __init__(self, start_time, stop_time, step):
        self.params = {}
        self.signal = []
        self.time = dict(start_time=start_time, stop_time=stop_time, step=step)

    def add_param(self, param, value):
        if param in self.params:
            print(f'ERROR: Параметр {param} уже существует')
        else:
            self.params[param] = value

    def edit_param(self, param, value):
        if param in self.params:
            self.params[param] = value
        else:
            print(f'ERROR: Параметр {param} отсутствует')

    def delete_param(self, param):
        if param in self.params:
            del self.params[param]
        else:
            print(f'ERROR: Параметр {param} отсутствует')

    def calc_signal(self):
        pass

    def print_signal(self):
        print(f'Current signal:\n {self.signal}')

    def graph_signal(self, title=''):
        if not any(self.signal):
            print(f'ERROR: Массив значений сигнала отсутствует')
        else:
            plt.plot(np.arange(self.time['start_time'],
                               self.time['stop_time'],
                               self.time['step']),
                     self.signal)
            if title:
                plt.title(title)
            plt.show()

    def fourier_transform(self, el_amount=None, axis=-1):
        return np.fft.fft(self.signal, el_amount, axis)

    def inverse_fourier_transform(self, el_amount=None, axis=-1):
        return np.fft.ifft(self.signal, el_amount, axis)

    @staticmethod
    def get_complex_plots(values, title=''):
        plt.plot(values.real, 'r-')
        real_title = title + ' Real part'
        plt.title(real_title)
        plt.show()
        plt.plot(values.imag, 'b-')
        imag_title = title + ' Imag part'
        plt.title(imag_title)
        plt.show()

    def get_plot_fourier_transform(self, title='', el_amount=None, axis=-1):
        transformed = self.fourier_transform(el_amount, axis)
        self.get_complex_plots(transformed, title=title)

    def get_plot_inverse_fourier_transform(self, title='', el_amount=None, axis=-1):
        transformed = self.inverse_fourier_transform(el_amount, axis)
        self.get_complex_plots(transformed, title=title)


class TSourceSignal(TSignal):

    def __init__(self, amplitude, frequency, phase,
                 start_time, stop_time, step):
        super().__init__(start_time, stop_time, step)
        self.add_param('amplitude', amplitude)
        self.add_param('frequency', frequency)
        self.add_param('phase', phase)


class TNoiseSignal(TSignal):

    def __init__(self, num_channel, energy, start_time, stop_time, step):
        super().__init__(start_time, stop_time, step)
        self.add_param('num_channel', num_channel)
        self.add_param('energy', energy)

    def get_bar_chart(self, x_start=None, x_end=None, y_start=None, y_end=None, patches=30, title=''):
        count, bins, ignored = plt.hist(self.signal, patches)
        bounds = x_start, x_end, y_start, y_end
        if all(bounds):
            plt.axis(list(bounds))
        plt.grid(True)
        plt.title(title)
        plt.show()
        return count, bins, ignored

    @staticmethod
    def get_density_distribution(density_distribution, title=''):
        plt.plot(density_distribution)
        plt.title(title)
        plt.show()

    @staticmethod
    def get_distribution_function(distrib_func, title=''):
        plt.plot(distrib_func)
        plt.title(title)
        plt.show()

    @property
    def expected_value(self):
        if 'expected_value' not in self.params:
            self.add_param('expected_value', np.mean(self.signal))
        return self.params['expected_value']

    @property
    def median(self):
        if 'median' not in self.params:
            self.add_param('median', np.median(self.signal))
        return self.params['median']

    @property
    def dispersion(self):
        if 'dispersion' not in self.params:
            self.add_param('dispersion', np.var(self.signal))
        return self.params['dispersion']

    @property
    def confidence_interval(self):
        if 'confidence_interval' not in self.params:
            confidence = 0.95
            a = 1.0 * np.array(self.signal)
            n = len(a)
            m, se = np.mean(a), sp_stats.sem(a)
            h = se * sp_stats.t.ppf((1 + confidence) / 2., n - 1)
            self.add_param('confidence_interval', {'median': m,
                                                   'left_bound': m - h,
                                                   'right_bound': m + h})
        return self.params['confidence_interval']

    def get_basic_random_distribution_params(self):
        print(f'Noise expected value: {self.expected_value}')
        print(f'Noise median: {self.median}')
        print(f'Noise dispersion: {self.dispersion}')
        print(f'Noise confidence interval: {self.confidence_interval}')
        return self.expected_value, self.median, self.dispersion, self.confidence_interval


class TConvertSignal(TSignal):
    def __init__(self, start_time, stop_time, step):
        super().__init__(start_time, stop_time, step)
        self.convert_params = dict()

    def add_convert_param(self, param, value):
        if param in self.convert_params:
            print(f'ERROR: Параметр {param} уже существует')
        else:
            self.convert_params[param] = value

    def edit_convert_param(self, param, value):
        if param in self.convert_params:
            self.convert_params[param] = value
        else:
            print(f'ERROR: Параметр {param} отсутствует')

    def delete_convert_param(self, param):
        if param in self.convert_params:
            del self.convert_params[param]
        else:
            print(f'ERROR: Параметр {param} отсутствует')


class TMixSignal(TConvertSignal):
    def __init__(self, first_signal, second_signal):
        super().__init__(first_signal.time['start_time'], first_signal.time['stop_time'],
                         first_signal.time['step'])
        self.first_signal = first_signal
        self.second_signal = second_signal

    def calc_signal(self):
        self.first_signal.calc_signal()
        self.second_signal.calc_signal()
        self.signal = self.first_signal.signal + self.second_signal.signal


class THFMSourceSignal(TSourceSignal):

    def __init__(self, amp, freq, end_freq, phase, start_time, stop_time, step):
        super().__init__(amp, freq, phase, start_time, stop_time, step)
        self.add_param('end_frequency', end_freq)

    def calc_signal(self):
        times = len(np.arange(self.time['start_time'],
                              self.time['stop_time'],
                              self.time['step']))
        sequence = np.linspace(self.time['start_time'], self.time['stop_time'], times)
        self.signal = chirp(self.params['amplitude'] * sequence,
                            f0=self.params['frequency'],
                            f1=self.params['end_frequency'],
                            t1=self.time['stop_time'] - self.time['start_time'], method='quadratic')


class TWeibNoiseSignal(TNoiseSignal):

    def __init__(self, num_channel, energy, start_time, stop_time, step, scale_factor, shape_factor):
        super().__init__(num_channel, energy, start_time, stop_time, step)
        self.add_param('scale_factor', scale_factor)
        self.add_param('shape_factor', shape_factor)

    def calc_signal(self):
        self.signal = self.weibull_distribution()

    def weibull_distribution(self):
        noise_time_arange = np.arange(self.time['start_time'], self.time['stop_time'], self.time['step'])
        weibull_distrib = np.random.weibull(self.params['shape_factor'], len(noise_time_arange))
        scaled_weib_distrib = weibull_distrib * self.params['scale_factor']
        return scaled_weib_distrib

    def density_distribution_weib(self):
        k = self.params['shape_factor']
        lamb = self.params['scale_factor']
        x = self.signal
        prob_dens = (k / lamb) * (x / lamb) ** (k - 1) * np.exp(-(x / lamb) ** k)
        return prob_dens

    def distribution_function_weib(self):
        k = self.params['shape_factor']
        lamb = self.params['scale_factor']
        res = [1 - np.exp(float((-x / lamb).real) ** k) for x in self.signal]  # only real values
        return res


class TChannel:

    def __init__(self, channel_len, channel_speed):
        self.channel_len = channel_len
        self.channel_speed = channel_speed

    def attenuate_signal(self, signal):
        pass

    @staticmethod
    def plot_attenuated_signal(attenuated, title='Attenuated'):
        plt.plot(attenuated)
        plt.title(title)
        plt.show()


class TLChannel(TChannel):  # L - linear

    def __init__(self, channel_len, channel_speed):
        super().__init__(channel_len, channel_speed)

    def attenuate_signal(self, signal):
        return signal / (self.channel_len * self.channel_speed)


if __name__ == '__main__':

    def process_thfm_signal():
        thfm_start = 0
        thfm_stop = 100
        thfm_step = 0.5

        thfm = THFMSourceSignal(0.5, 20, 25, 0, thfm_start, thfm_stop, thfm_step)
        thfm.calc_signal()
        thfm.graph_signal(title='THFM signal')
        thfm.get_plot_fourier_transform(title='THFM Fourier transform')
        thfm.get_plot_inverse_fourier_transform(title='THFM Inverse Fourier transform')

    def process_weibull_noise(noise_start=None,
                              noise_stop=None,
                              noise_step=None,
                              noise_num_ch=None,
                              noise_energy=None,
                              scale_factor=None,
                              shape_factor=None):

        noise_start = 0 if not noise_start else noise_start
        noise_stop = 9 * np.pi / 4 if not noise_stop else noise_stop
        noise_step = np.pi / 4 if not noise_step else noise_step
        noise_num_ch = 1 if not noise_num_ch else noise_num_ch
        noise_energy = 1 if not noise_energy else noise_energy
        scale_factor = 4 if not scale_factor else scale_factor  # lambda
        shape_factor = 3 if not shape_factor else shape_factor  # k

        noise = TWeibNoiseSignal(noise_num_ch,
                                 noise_energy,
                                 noise_start,
                                 noise_stop,
                                 noise_step,
                                 scale_factor,
                                 shape_factor)

        noise.calc_signal()

        noise.graph_signal(title='Weibull noise signal')

        noise.get_bar_chart(title='Weibull distribution bar chart')

        density_distribution = noise.density_distribution_weib()
        noise.get_density_distribution(density_distribution, title='Weibull density distribution')

        distribution_function = noise.distribution_function_weib()
        noise.get_distribution_function(distribution_function, title='Weibull distribution function')

        noise.get_plot_fourier_transform(title='Weibull noise Fourier transform')

        noise.get_plot_inverse_fourier_transform(title='THFM Inverse Fourier transform')

        noise.get_basic_random_distribution_params()

    def mix_signals():
        thfm_start = 0
        thfm_stop = 100
        thfm_step = 0.5

        thfm = THFMSourceSignal(0.5, 20, 25, 0, thfm_start, thfm_stop, thfm_step)
        thfm.calc_signal()
        thfm.graph_signal(title='THFM to mix')
        noise_start = 0
        noise_stop = 200 * np.pi / 4
        noise_step = np.pi / 4
        noise_num_ch = 1
        noise_energy = 1
        scale_factor = 4  # lambda
        shape_factor = 3  # k

        noise = TWeibNoiseSignal(noise_num_ch,
                                 noise_energy,
                                 noise_start,
                                 noise_stop,
                                 noise_step,
                                 scale_factor,
                                 shape_factor)
        noise.calc_signal()
        noise.graph_signal(title='Weibull noise for THFM')
        basic_signal = thfm
        noise_signal = noise
        mixed = TMixSignal(basic_signal, noise_signal)
        mixed.calc_signal()
        mixed.graph_signal(title='Mixed signal (THFM + Weibull noise)')
        mixed.get_plot_fourier_transform(title='Mixed (THFM + Weibull noise) Fourier transform')
        mixed.get_plot_inverse_fourier_transform(title='Mixed (THFM + Weibull noise) Inverse Fourier transform')

    def attenuate_signal():
        t_start = 0
        t_stop = 9 * np.pi / 4
        t_step = np.pi / 4

        t_time_arange = np.arange(t_start, t_stop, t_step)
        sin_to_attenuate = np.sin(t_time_arange)
        plt.plot(sin_to_attenuate)
        plt.title('sin to attenuate')
        plt.show()
        channel = TLChannel(3, 330)
        attenuated = channel.attenuate_signal(sin_to_attenuate)
        channel.plot_attenuated_signal(attenuated=attenuated, title='attenuated sin')

    process_thfm_signal()
    process_weibull_noise()
    mix_signals()
    attenuate_signal()
