
from pathlib import Path
from rich.console import Console
from absl import app, flags
from absl.flags import FLAGS
import pandas as pd
from tqdm import tqdm
import os
from itertools import  repeat
from multiprocessing.dummy import Pool


flags.DEFINE_string('d', 'images', 'Path to vggface2 dataset')
flags.DEFINE_string('c', './dataset.csv', 'Path to save the csv file')


def write_dir_imgs(dir : str, mapping : dict):
    '''
    takes in a path to identity dir. and writes the csv file in tmp folder

    Parameters
    ----------
    dir : str
        identity dir path which contains images from the same identity
    mapping : dict
        map identity string to a specific unique index integer
    ''' 
        
    assert Path("tmp").exists() , "tmp folder does not exist"
    imgs = [k for k in Path(FLAGS.d , dir).iterdir() if k.is_file()]
    # Console().print(imgs)
    d = {}
    for k in tqdm(imgs, colour="green", leave=False):
        d.setdefault("image", []).append(str(k))
        d.setdefault("identity", []).append(str(k.parents[0].stem))
        d.setdefault("label", []).append(mapping[str(k.parents[0].stem)])
    # save to csv with specific identity name.
    pd.DataFrame.from_dict(d).to_csv(f"./tmp/{dir}.csv", index=False)
    
def main(argv):
    
    # get all identities strings
    identities = os.listdir(FLAGS.d)    
    # idn to label mapping
    idn2idx = {v:k for k,v in enumerate(identities)}
    
    with Pool(8) as p:
        
        p.starmap(write_dir_imgs, zip(identities, repeat(idn2idx)))

    Console().rule(title="DONE", style="bold green", characters="=")

if __name__ == '__main__':
    Path("tmp").mkdir(parents=True, exist_ok=True)
    app.run(main)