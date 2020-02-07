fn = 'TOX21/NR-AR_wholetraining.smiles'
f = open(fn, "r")
ffake = open("NR-AR_fakelabels", "w")
lines = f.readlines()
for line in lines:
    splitted = line.split(" ")
    ffake.write(splitted[0] + " 0\n")

f.close()
ffake.close()

