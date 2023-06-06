import os, shutil

IMNET21K_DIR = sys.argv[1]
IMNET2K_DIR = sys.argv[2]
ORDER_FILE_DIR = sys.argv[3]

f = open(ORDER_FILE_DIR+'/class_order.txt', 'r')
lines = f.readlines()

for line in lines:
    os.makedirs(IMNET2K_DIR+'train/'+line.strip(), exist_ok=True)
    os.makedirs(IMNET2K_DIR+'val/'+line.strip(), exist_ok=True)
    os.makedirs(IMNET2K_DIR+'test/'+line.strip(), exist_ok=True)

f = open(ORDER_FILE_DIR+'/val.txt', 'r')
lines = f.readlines()

for line in lines:
    line = line.strip()
    to = IMNET2K_DIR+line[1:]
    prev = IMNET21K_DIR+line.split('/')[2]+'/'+line.split('/')[3]
    shutil.copy(IMNET21K_DIR+line.split('/')[2]+'/'+line.split('/')[3], IMNET2K_DIR+line[1:])

f = open(ORDER_FILE_DIR+'/test.txt', 'r')
lines = f.readlines()

for line in lines:
    line = line.strip()
    to = IMNET2K_DIR+line[1:]
    prev = IMNET21K_DIR+line.split('/')[2]+'/'+line.split('/')[3]
    shutil.copy(IMNET21K_DIR+line.split('/')[2]+'/'+line.split('/')[3], IMNET2K_DIR+line[1:])

f = open(ORDER_FILE_DIR+'/class_incremental_ordering.txt', 'r')
lines = f.readlines()

for line in lines:
    line = line.strip()
    to = IMNET2K_DIR+line[1:]
    prev = IMNET21K_DIR+line.split('/')[2]+'/'+line.split('/')[3]
    shutil.copy(IMNET21K_DIR+line.split('/')[2]+'/'+line.split('/')[3], IMNET2K_DIR+line[1:])
