getSymbols.yahoo("SUNPHARMA.NS",env = parent.frame(),from = as.Date("2018-05-01"),to = as.Date("2019-05-01"))
sunpharma = SUNPHARMA.NS
sun_fft = fft(sunpharma$`Close Price`[1:45])
sun_mod = Mod(sun_fft)/length(sun_fft)
sun_mod2<-(sun_mod[1:(length(sun_mod))])
sun_noi = c()
i = c(1:45)
for(j in i){
  if(sun_mod2[j] > 3 && j<45){
    sun_noi = c(sun_noi,sun_fft[j])
  }else{
    sun_noi = c(sun_noi,0)
  }
}
plot(detrend(sunpharma$`Close Price`[1:45]),type='l')
lines(ind[1:45],Mod(ifft(sun_noi)),type='l',col='red')

