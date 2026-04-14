"""
Wav2Lip model architecture.
Based on: https://github.com/Rudrabha/Wav2Lip (MIT License)
Prajwal et al., "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild", ACM MM 2020
"""
import torch
from torch import nn


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(
                cin, cout, kernel_size,
                stride=stride, padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_block(x))


class Wav2Lip(nn.Module):
    """
    Wav2Lip: takes a face image (masked+original, 6 ch) and a mel spectrogram
    and outputs a synthesised face with accurate lip sync.

    Input:
        audio_sequences : (B, 1, 80, 16)   mel spectrogram chunk
        face_sequences  : (B, 6, H, W)     [masked_face | original_face], H=W=96
    Output:
        (B, 3, H, W)  synthesised face in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # ── Face encoder (down-sampling) ──────────────────────────────────────
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, 7, 1, 3)),                              # 96×96
            nn.Sequential(
                Conv2d(16, 32, 3, 2, 1),                                        # 48×48
                Conv2d(32, 32, 3, 1, 1, residual=True),
                Conv2d(32, 32, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(32, 64, 3, 2, 1),                                        # 24×24
                Conv2d(64, 64, 3, 1, 1, residual=True),
                Conv2d(64, 64, 3, 1, 1, residual=True),
                Conv2d(64, 64, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(64, 128, 3, 2, 1),                                       # 12×12
                Conv2d(128, 128, 3, 1, 1, residual=True),
                Conv2d(128, 128, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(128, 256, 3, 2, 1),                                      #  6×6
                Conv2d(256, 256, 3, 1, 1, residual=True),
                Conv2d(256, 256, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(256, 512, 3, 2, 1),                                      #  3×3
                Conv2d(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(512, 512, 3, 1, 0),                                      #  1×1
                Conv2d(512, 512, 1, 1, 0),
            ),
        ])

        # ── Audio encoder ─────────────────────────────────────────────────────
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1, residual=True),
            Conv2d(32, 32, 3, 1, 1, residual=True),

            Conv2d(32, 64, 3, (3, 1), 1),
            Conv2d(64, 64, 3, 1, 1, residual=True),
            Conv2d(64, 64, 3, 1, 1, residual=True),

            Conv2d(64, 128, 3, 3, 1),
            Conv2d(128, 128, 3, 1, 1, residual=True),
            Conv2d(128, 128, 3, 1, 1, residual=True),

            Conv2d(128, 256, 3, (3, 2), 1),
            Conv2d(256, 256, 3, 1, 1, residual=True),

            Conv2d(256, 512, 3, 1, 0),
            Conv2d(512, 512, 1, 1, 0),
        )

        # ── Face decoder (up-sampling with skip connections) ──────────────────
        #
        # Channel accounting (face_encoder feats channels are listed on the right):
        #  Block  Input ch   Output ch  feats popped  post-concat ch
        #  ─────  ─────────  ─────────  ────────────  ──────────────
        #   0     512        512        512            1024
        #   1     1024       512        512            1024
        #   2     1024       512        256            768
        #   3     768        384        128            512
        #   4     512        256        64             320
        #   5     320        128        32             160
        #   6     160        64         16             80
        #  out    80         3          —              —
        #
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, 1, 1, 0)),                           #  1×1
            nn.Sequential(
                Conv2dTranspose(1024, 512, 3, 1, 0),                            #  3×3
                Conv2d(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(1024, 512, 3, 2, 1, output_padding=1),         #  6×6
                Conv2d(512, 512, 3, 1, 1, residual=True),
                Conv2d(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(768, 384, 3, 2, 1, output_padding=1),          # 12×12
                Conv2d(384, 384, 3, 1, 1, residual=True),
                Conv2d(384, 384, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(512, 256, 3, 2, 1, output_padding=1),          # 24×24
                Conv2d(256, 256, 3, 1, 1, residual=True),
                Conv2d(256, 256, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(320, 128, 3, 2, 1, output_padding=1),          # 48×48
                Conv2d(128, 128, 3, 1, 1, residual=True),
                Conv2d(128, 128, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(160, 64, 3, 2, 1, output_padding=1),           # 96×96
                Conv2d(64, 64, 3, 1, 1, residual=True),
                Conv2d(64, 64, 3, 1, 1, residual=True),
            ),
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, 3, 1, 1),
            nn.Conv2d(32, 3, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, audio_sequences: torch.Tensor, face_sequences: torch.Tensor) -> torch.Tensor:
        """
        audio_sequences : (B, 1, 80, 16)
        face_sequences  : (B, 6, H, W)
        returns         : (B, 3, H, W)  in [0, 1]
        """
        audio_embedding = self.audio_encoder(audio_sequences)  # (B, 512, 1, 1)

        feats: list[torch.Tensor] = []
        x = face_sequences
        for block in self.face_encoder_blocks:
            x = block(x)
            feats.append(x)

        x = audio_embedding
        for block in self.face_decoder_blocks:
            x = block(x)
            if feats:
                skip = feats.pop()
                if x.shape == skip.shape:
                    x = torch.cat((x, skip), dim=1)
                else:
                    # Spatial mismatch guard (shouldn't happen with correct input sizes)
                    x = torch.cat((x, skip[:, :, : x.shape[2], : x.shape[3]]), dim=1)

        return self.output_block(x)


def load_wav2lip(checkpoint_path: str, device: str = "cuda") -> Wav2Lip:
    """Load Wav2Lip from a .pth checkpoint file."""
    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Strip 'model.' prefix if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
