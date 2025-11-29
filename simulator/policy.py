from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class WordleMLPPolicy(nn.Module):
    """
    Policy πθ(a | s) for Wordle.

    Inputs:
      - obs:        [B, 26, 5, 3]  (from WordleEnv)
      - guess_count: [B] integers in [0, max_guesses-1]

    Internally:
      - MLP maps (obs_flat + guess_onehot) -> [B, 26, 5] value matrix V

    Word scoring:
      - Precomputed word_features: [num_words, 26, 5]
      - For each batch element, logits = <V_flat, word_features_flat> over all words
    """

    def __init__(
        self,
        word_features: torch.Tensor,
        max_guesses: int,
        hidden_dim: int = 256,
    ):
        """
        word_features: [num_words, 26, 5] tensor from build_word_features(...)
        max_guesses:   maximum number of guesses in the env (e.g., 6)
        hidden_dim:    size of hidden layer in the MLP
        """
        super().__init__()

        assert word_features.ndim == 3
        num_words, L, P = word_features.shape
        assert L == 26 and P == 5, "word_features must be [num_words, 26, 5]"

        self.num_words = num_words
        self.max_guesses = max_guesses

        # Register word_features as buffers so they move with .to(device)
        self.register_buffer("word_features", word_features)  # [W, 26, 5]
        self.register_buffer(
            "word_features_flat",
            word_features.view(num_words, -1),                # [W, 130]
        )

        # Input dimensions
        obs_dim = 26 * 5 * 3               # 390
        guess_feat_dim = max_guesses       # one-hot encoding
        in_dim = obs_dim + guess_feat_dim  # total input to MLP

        out_dim = 26 * 5                   # flattened V matrix

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)


    def save(self, path: str) -> None:
        """
        Save the model parameters to a file.
        Use: policy.save("checkpoints/wordle_policy.pt")
        """
        torch.save(self.state_dict(), path)

    # --- NEW: load from file as a convenience constructor ---
    @classmethod
    def load_from_file(
        cls,
        path: str,
        word_features: torch.Tensor,
        max_guesses: int,
        hidden_dim: int = 256,
        map_location: str | torch.device = "cpu",
    ) -> "WordleMLPPolicy":
        """
        Create a WordleMLPPolicy instance and load weights from a checkpoint file.

        Args:
            path: path to a .pt/.pth file saved via policy.save(...)
            word_features: [num_words, 26, 5] tensor (same as in __init__)
            max_guesses: max guesses in env (e.g., 6)
            hidden_dim: hidden layer size (must match the one used when saving)
            map_location: device to map weights to ("cpu" or torch.device("cuda"))
        """
        model = cls(word_features=word_features, max_guesses=max_guesses, hidden_dim=hidden_dim)
        state_dict = torch.load(path, map_location=map_location)
        model.load_state_dict(state_dict)
        return model

    def forward(
        self,
        obs: torch.Tensor,
        guess_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to produce the letter-position value matrix V.

        obs:         [B, 26, 5, 3], dtype float or convertible to float
        guess_count: [B], dtype long/int with values in [0, max_guesses-1]

        Returns:
            V: [B, 26, 5]
        """
        if obs.ndim == 3:
            # allow passing a single obs [26, 5, 3]
            obs = obs.unsqueeze(0)   # [1, 26, 5, 3]
        if guess_count.ndim == 0:
            guess_count = guess_count.unsqueeze(0)  # [1]

        B = obs.size(0)

        # Flatten obs: [B, 26*5*3]
        obs_flat = obs.view(B, -1).float()

        # One-hot encode guess_count: [B, max_guesses]
        guess_onehot = F.one_hot(
            guess_count.long(),
            num_classes=self.max_guesses,
        ).float()

        x = torch.cat([obs_flat, guess_onehot], dim=-1)  # [B, in_dim]

        h = F.relu(self.fc1(x))
        V_flat = self.fc2(h)                             # [B, 130]
        V = V_flat.view(B, 26, 5)                        # [B, 26, 5]

        return V

    def get_logits(
        self,
        obs: torch.Tensor,
        guess_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits over the candidate words.

        Returns:
            logits: [B, num_words]
        """
        V = self.forward(obs, guess_count)         # [B, 26, 5]
        V_flat = V.view(V.size(0), -1)             # [B, 130]

        # word_features_flat: [num_words, 130]
        # logits = V_flat @ word_features_flat^T -> [B, num_words]
        logits = V_flat @ self.word_features_flat.t()
        return logits

    def get_distribution(
        self,
        obs: torch.Tensor,
        guess_count: torch.Tensor,
    ) -> Categorical:
        """
        Build a Categorical distribution over word indices.
        """
        logits = self.get_logits(obs, guess_count)    # [B, num_words]
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        guess_count: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method:
          - returns (action_indices, log_probs)
        obs:         [26, 5, 3] or [B, 26, 5, 3]
        guess_count: scalar or [B]
        """
        dist = self.get_distribution(obs, guess_count)

        if greedy:
            # argmax policy
            logits = dist.logits
            action = torch.argmax(logits, dim=-1)
        else:
            # sample from the policy
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob
