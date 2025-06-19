import numpy as np


class SubmodularFunction(object):
    def __init__(
        self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]
    ):
        self.index = index
        self.n = len(index)

        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None

        # For the sample similarity matrix, the method supports two input modes, one is to input a pairwise similarity
        # matrix for the whole sample, and the other case allows the input of a similarity kernel to be used to
        # calculate similarities incrementally at a later time if required.
        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert (
                similarity_matrix.shape[0] == self.n
                and similarity_matrix.shape[1] == self.n
            )
            self.similarity_matrix = similarity_matrix
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

    def _similarity_kernel(self, similarity_kernel):
        return similarity_kernel


class FacilityLocation(SubmodularFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.already_selected.__len__() == 0:
            self.cur_max = np.zeros(self.n, dtype=np.float32)
        else:
            self.cur_max = np.max(
                self.similarity_kernel(np.arange(self.n), self.already_selected), axis=1
            )

        self.all_idx = np.ones(self.n, dtype=bool)

    def _similarity_kernel(self, similarity_kernel):
        # Initialize a matrix to store similarity values of sample points.
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(
                    self.all_idx, not_calculated
                )
                self.if_columns_calculated[not_calculated] = True
            return self.sim_matrix[np.ix_(a, b)]

        return _func

    def calc_gain(self, idx_gain, selected, **kwargs):
        gains = np.maximum(
            0.0,
            self.similarity_kernel(self.all_idx, idx_gain)
            - self.cur_max.reshape(-1, 1),
        ).sum(axis=0)
        return gains

    def calc_gain_batch(self, idx_gain, selected, **kwargs):
        batch_idx = ~self.all_idx
        batch_idx[0 : kwargs["batch"]] = True
        gains = np.maximum(
            0.0,
            self.similarity_kernel(batch_idx, idx_gain)
            - self.cur_max[batch_idx].reshape(-1, 1),
        ).sum(axis=0)
        for i in range(kwargs["batch"], self.n, kwargs["batch"]):
            batch_idx = ~self.all_idx
            batch_idx[i * kwargs["batch"] : (i + 1) * kwargs["batch"]] = True
            gains += np.maximum(
                0.0,
                self.similarity_kernel(batch_idx, idx_gain)
                - self.cur_max[batch_idx].reshape(-1, 1),
            ).sum(axis=0)
        return gains

    def update_state(self, new_selection, total_selected, **kwargs):
        self.cur_max = np.maximum(
            self.cur_max,
            np.max(self.similarity_kernel(self.all_idx, new_selection), axis=1),
        )
