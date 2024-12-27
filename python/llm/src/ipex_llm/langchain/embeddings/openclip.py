#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

# This file is adapted from
# https://github.com/langchain-ai/langchain/blob/langchain-experimental%3D%3D0.0.62/libs/experimental/langchain_experimental/open_clip/open_clip.py

# MIT License

# Copyright (c) LangChain, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, Dict, List

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain_core.embeddings import Embeddings


class OpenCLIPEmbeddings(BaseModel, Embeddings):
    """OpenCLIP Embeddings model."""

    model: Any
    preprocess: Any
    tokenizer: Any
    # Select model: https://github.com/mlfoundations/open_clip
    model_name: str = "ViT-H-14"
    checkpoint: str = "laion2b_s32b_b79k"
    device: str = "cpu"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip and torch libraries are installed."""
        try:
            import open_clip

            # Fall back to class defaults if not provided
            model_name = values.get("model_name", cls.__fields__["model_name"].default)
            checkpoint = values.get("checkpoint", cls.__fields__["checkpoint"].default)
            device = values.get("device", cls.__fields__["device"].default)

            if device not in ["cpu", "xpu"] and not device.startswith("xpu:"):
              raise ValueError(
                  "OpenCLIPEmbeddings currently only supports device to be 'cpu', 'xpu', "
                  f"or 'xpu:<device_id>', but you have: {device}."
              )

            # Load model
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name, pretrained=checkpoint
            )

            if device == 'xpu' or device.startswith("xpu:"):
                model = model.to(device)

            tokenizer = open_clip.get_tokenizer(model_name)
            values["model"] = model
            values["preprocess"] = preprocess
            values["tokenizer"] = tokenizer
            values["device"] = device

        except ImportError:
            raise ImportError(
                "Please ensure both open_clip and torch libraries are installed. "
                "pip install open_clip_torch torch"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_features = []
        for text in texts:
            # Tokenize the text
            tokenized_text = self.tokenizer(text)

            if self.device == 'xpu' or self.device.startswith("xpu:"):
                tokenized_text = tokenized_text.to(self.device)

            # Encode the text to get the embeddings
            embeddings_tensor = self.model.encode_text(tokenized_text).cpu()

            # Normalize the embeddings
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert normalized tensor to list and add to the text_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            text_features.append(embeddings_list)

        return text_features

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")

        # Open images directly as PIL images
        pil_images = [_PILImage.open(uri) for uri in uris]

        image_features = []
        for pil_image in pil_images:
            # Preprocess the image for the model
            preprocessed_image = self.preprocess(pil_image).unsqueeze(0)

            if self.device == 'xpu' or self.device.startswith("xpu:"):
                 preprocessed_image = preprocessed_image.to(self.device)

            # Encode the image to get the embeddings
            embeddings_tensor = self.model.encode_image(preprocessed_image).cpu()

            # Normalize the embeddings tensor
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert tensor to list and add to the image_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()

            image_features.append(embeddings_list)

        return image_features
