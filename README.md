# Textual_TimeTravel_TOM

This project contails the code for training heuristic and learned time travel models. 

# Model set up

The model set up and the code is based on EntNet [1] (https://github.com/facebookarchive/MemNN/tree/master/EntNet-babi).

## Usage

To train a model, run the following command

    th main.lua

# Corrected ToMi Dataset

The code corrects the errors in second order questions of the ToMi dataset [2] (https://github.com/facebookresearch/ToMi). To run:

    python main.py


## References

*[1] Mikael Henaff, Jason Weston, Arthur Szlam, Antoine Bordes, and Yann LeCun, "[Tracking the World State with Recurrent Entity Networks](https://arxiv.org/abs/1612.03969)", *arXiv:1612.03969 [cs.CL]*.

*[2] Le, Matthew  and Boureau, Y-Lan and Nickel, Maximilian, "Revisiting the Evaluation of Theory of Mind through Question Answering (https://www.aclweb.org/anthology/D19-1598) *.