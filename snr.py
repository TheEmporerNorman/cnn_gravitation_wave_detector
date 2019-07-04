
def SNR(HF ,lstH_inj, H_back):
    data=np.array(HF)
    temp=np.array(lstH_inj)
    back=np.array(H_back)

    data_fft=np.fft.fft(data)
    template_fft = np.fft.fft(temp)

    # -- Calculate the PSD of the data
    power_data, freq_psd = plt.psd(back, Fs=fs, NFFT=fs, visible=False)

    # -- Interpolate to get the PSD values at the needed frequencies
    datafreq = np.fft.fftfreq(data.size)*fs
    power_vec = np.interp(datafreq, freq_psd, power_data)
    # -- Calculate the matched filter output
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)
    # -- Normalize the matched filter output
    df = np.abs(datafreq[1] - datafreq[0])
    sigmasq = 2*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = abs(optimal_time) / (sigma)

    #return(max(SNR))