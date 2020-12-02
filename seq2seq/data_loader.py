import numpy as np
import torch
import torch.utils.data as data


class S2SDataset(data.Dataset):
    def __init__(self, protein_data, ids):

        data_len = len(ids)

        # data_len, 700, 22 one hot
        all_encodings = np.zeros([data_len, 700, 22])
        
        # data_len, 700 x 21 PSSM
        all_pssm = np.zeros([data_len, 700, 21])
        all_lengths = []

        for i, id in enumerate(ids):
            id = str(id)
            if i % 250 == 0:
                print("Loading {0}/{1} proteins".format(i, len(ids)))

            d = protein_data[id]
            protein_length = d["protein_length"]
            all_lengths.append(protein_length)
            
            reshaped = np.array(d["protein_encoding"]).reshape([700, -1])

            all_encodings[i, :] = reshaped[:, 0:22]
            all_pssm[i, :] = reshaped[:, 29:50]

        self.all_encodings = all_encodings.astype(np.uint8)
        self.all_pssm = all_pssm.astype(np.float32)
        self.all_lengths = np.array(all_lengths).astype(np.int32)

        print(len(all_pssm), len(all_pssm), len(all_lengths))

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        encoding = self.all_encodings[index]
        pssm = self.all_pssm[index]
        length = self.all_lengths[index]

        return encoding, pssm, length

    def __len__(self):
        return len(self.all_encodings)


def get_loader(protein_data, ids, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader"""

    protein = S2SDataset(protein_data, ids)

    # def collate_fn(data):
    #     return data

    data_loader = torch.utils.data.DataLoader(dataset=protein,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers, )
    # collate_fn=collate_fn)
    return data_loader, len(protein)