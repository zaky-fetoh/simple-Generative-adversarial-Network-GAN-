import train as tr
import model as md
import utility as util


gan, discr = md.compile_GAN()
tr.train([gan,discr], 1000,256);
gan.save('gan_model.h5')
util.plot(gan.layers[0])