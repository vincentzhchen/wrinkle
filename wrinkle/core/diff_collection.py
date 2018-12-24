import collections


class DiffCollection(object):
    def __init__(self, lhs, rhs, ignore_duplicates=True):
        self.lhs = lhs
        self.rhs = rhs

        # collections of hashable items only
        if not self._is_all_hashable(
                self.lhs) or not self._is_all_hashable(self.rhs):
            raise ValueError("COLLECTIONS MUST contain hashable elements.")

        if not ignore_duplicates:
            self._diff_iterable()
        else:
            self.lhs = set(self.lhs)
            self.rhs = set(self.rhs)
            self._diff_as_set()

    def _is_all_hashable(self, arg):
        return all(isinstance(x, collections.Hashable) for x in arg)

    def _diff_as_set(self):
        """Sets have hashable elements, so use set methods.

        """
        self.in_lhs = self.lhs.difference(self.rhs)
        self.in_rhs = self.rhs.difference(self.lhs)
        self.symmetric_difference = self.lhs.symmetric_difference(self.rhs)

    def _diff_iterable(self):
        """This is not efficient but keeps track of duplicates.

        """
        self.in_lhs = [x for x in self.lhs if x not in self.rhs]
        self.in_rhs = [x for x in self.rhs if x not in self.lhs]
        self.symmetric_difference = self.in_lhs + self.in_rhs

    def get_values_only_in_lhs(self):
        return self.in_lhs

    def get_values_only_in_rhs(self):
        return self.in_rhs
