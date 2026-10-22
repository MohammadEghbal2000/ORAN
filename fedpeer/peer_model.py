"""Offers-only local peer model theta_{s->s'} (proposal eqs. 7-8) with the bid loss used by B2."""
import numpy as np

from .contracts import ROLES, XN, oriented

# raw parameters z = [w logits (3), log gamma (3), reservation logits (R, D), log tau]; Z0 is the uninformed prior
Z0 = np.array([0, 0, 0, 0, 0, 0, np.log(0.35 / 0.65), np.log(0.35 / 0.65), np.log(0.1)])


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _num_grad(f, z, h=1e-4):
    # shortcut: central differences over 9 parameters; switch to autodiff if the model grows
    g = np.empty_like(z)
    for i in range(len(z)):
        e = np.zeros_like(z)
        e[i] = h
        g[i] = (f(z + e) - f(z - e)) / (2 * h)
    return g


class PeerModel:
    """Persistent estimate of one counterpart: weights w, shapes gamma, reservation r per role, temperature tau."""

    def __init__(self, prior_strength=1.0, concession_weight=5.0, lr=0.05, steps=30, window=400):
        self.z = Z0.copy()
        self.prior_strength, self.concession_weight = prior_strength, concession_weight
        self.lr, self.steps, self.window = lr, steps, window
        self.obs = []          # (contract index, counterpart role, 1 if the counterpart would accept)
        self.concessions = []  # (earlier, later) counterpart offers, counterpart role
        self.n_negotiations = 0  # D_n, the local evidence count used later by federation

    @staticmethod
    def unpack(z):
        w = np.exp(z[:3]) / np.exp(z[:3]).sum()
        return w, np.exp(z[3:6]), _sigmoid(z[6:8]), np.exp(z[8])

    def utility(self, role):
        w, g, _, _ = self.unpack(self.z)
        return (w * oriented(XN, role) ** g).sum(1)

    def p_accept(self, role):
        _, _, r, tau = self.unpack(self.z)
        return _sigmoid((self.utility(role) - r[ROLES.index(role)]) / tau)

    def observe(self, history, counterpart_role, agreed):
        """history: [(proposer role, contract index)]. Their offers and an accepted offer of ours -> 1, our rejected offers -> 0."""
        last = len(history) - 1
        for k, (p, i) in enumerate(history):
            self.obs.append((i, counterpart_role, 1.0 if p == counterpart_role or (agreed and k == last) else 0.0))
        theirs = [i for p, i in history if p == counterpart_role]
        self.concessions += [(a, b, counterpart_role) for a, b in zip(theirs, theirs[1:])]
        self.obs, self.concessions = self.obs[-self.window:], self.concessions[-self.window:]
        self.n_negotiations += 1
        self._fit()

    def _data(self):
        data = []
        for k, role in enumerate(ROLES):
            o = oriented(XN, role)
            obs = [(i, y) for i, r, y in self.obs if r == role]
            con = [(a, b) for a, b, r in self.concessions if r == role]
            idx = np.array([i for i, _ in obs], int)
            data.append((k, o[idx], np.array([y for _, y in obs]),
                         o[np.array([a for a, _ in con], int)], o[np.array([b for _, b in con], int)]))
        return data

    def _loss(self, z, data):
        w, g, r, tau = self.unpack(z)
        total = self.prior_strength * np.sum((z - Z0) ** 2)
        for k, xo, y, xa, xb in data:
            p = np.clip(_sigmoid(((w * xo ** g).sum(1) - r[k]) / tau), 1e-6, 1 - 1e-6)
            total -= np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            # the counterpart's later offers should not be better for it than its earlier ones
            total += self.concession_weight * np.maximum(0, (w * xb ** g).sum(1) - (w * xa ** g).sum(1)).sum()
        return total

    def _fit(self):
        data = self._data()
        m = v = np.zeros_like(self.z)
        for t in range(1, self.steps + 1):  # Adam, warm-started from the persistent estimate
            g = _num_grad(lambda z: self._loss(z, data), self.z)
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            self.z = self.z - self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # evaluator-only error metrics against synthetic ground truth
    def weight_mae(self, true_w):
        return float(np.abs(self.unpack(self.z)[0] - true_w).mean())

    def utility_mae(self, true_prefs, role):
        return float(np.abs(self.utility(role) - true_prefs.utility(role)).mean())
