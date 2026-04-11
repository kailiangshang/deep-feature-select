from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deepfs import (
    ConcreteAutoencoderModel,
    GumbelSigmoidGateModel,
    GumbelSoftmaxGateConcreteModel,
    GumbelSoftmaxGateModel,
    GumbelSoftmaxGateIndirectConcreteModel,
    HardConcreteGateConcreteModel,
    HardConcreteGateModel,
    HardConcreteGateIndirectConcreteModel,
    IndirectConcreteAutoencoderModel,
    StochasticGateConcreteModel,
    StochasticGateModel,
    StochasticGateIndirectConcreteModel,
)
from deepfs.core.types import SparsityLoss

INPUT_DIM = 100
K = 10
ENCODER_EMB = 16
GATE_EMB = 8
BATCH_SIZE = 4
EPOCHS = 3


class _SimpleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def _make_data(batch_size=BATCH_SIZE, input_dim=INPUT_DIM, num_classes=5):
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


class TestEncoderModels:
    @pytest.fixture
    def data(self):
        return _make_data()

    def test_cae_train_eval(self, data):
        x, _ = data
        model = ConcreteAutoencoderModel(INPUT_DIM, K, total_epochs=EPOCHS)
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        result = model.get_selection_result()
        assert result.num_selected <= K

    def test_cae_temperature_update(self, data):
        x, _ = data
        model = ConcreteAutoencoderModel(INPUT_DIM, K, total_epochs=EPOCHS)
        model.train()
        model(x)
        old_temp = model.temperature.item()
        model.update_temperature(EPOCHS + 1)
        assert model.temperature.item() < old_temp

    def test_cae_encoder_diagnostics(self, data):
        x, _ = data
        model = ConcreteAutoencoderModel(INPUT_DIM, K, total_epochs=EPOCHS)
        model.train()
        model(x)
        diag = model.encoder_diagnostics()
        assert diag.selected_indices is not None
        assert len(diag.selected_indices) == K

    def test_ipcae_train_eval(self, data):
        x, _ = data
        model = IndirectConcreteAutoencoderModel(
            INPUT_DIM, K, embedding_dim=ENCODER_EMB, total_epochs=EPOCHS
        )
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)


class TestGateModels:
    @pytest.fixture
    def data(self):
        return _make_data()

    def test_stg_train_eval_sparsity(self, data):
        x, _ = data
        model = StochasticGateModel(INPUT_DIM, sigma=0.5)
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)
        sparsity = model.sparsity_loss()
        assert isinstance(sparsity, SparsityLoss)
        assert sparsity.total.requires_grad
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_gsg_sigmoid_train_eval_sparsity(self, data):
        x, _ = data
        model = GumbelSigmoidGateModel(INPUT_DIM, total_epochs=EPOCHS)
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)
        sparsity = model.sparsity_loss()
        assert sparsity.total.requires_grad
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_gsg_softmax_train_eval_sparsity(self, data):
        x, _ = data
        model = GumbelSoftmaxGateModel(INPUT_DIM, embedding_dim=GATE_EMB, total_epochs=EPOCHS)
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)
        sparsity = model.sparsity_loss()
        assert sparsity.total.requires_grad
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_hcg_train_eval_sparsity(self, data):
        x, _ = data
        model = HardConcreteGateModel(INPUT_DIM)
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)
        sparsity = model.sparsity_loss()
        assert sparsity.total.requires_grad
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_gate_diagnostics(self, data):
        x, _ = data
        model = StochasticGateModel(INPUT_DIM)
        model.train()
        model(x)
        diag = model.gate_diagnostics()
        assert diag.gate_probs is not None
        assert 0.0 <= diag.open_ratio <= 1.0


class TestCombinedModels:
    @pytest.fixture
    def data(self):
        return _make_data()

    def test_gsg_softmax_cae(self, data):
        x, _ = data
        model = GumbelSoftmaxGateConcreteModel(
            INPUT_DIM, K, embedding_dim_gate=GATE_EMB, total_epochs=EPOCHS
        )
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        sparsity = model.sparsity_loss()
        assert sparsity.total.requires_grad
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)

    def test_gsg_softmax_ipcae(self, data):
        x, _ = data
        model = GumbelSoftmaxGateIndirectConcreteModel(
            INPUT_DIM, K, embedding_dim_encoder=ENCODER_EMB,
            embedding_dim_gate=GATE_EMB, total_epochs=EPOCHS
        )
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        sparsity = model.sparsity_loss()
        assert sparsity.total.requires_grad
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        diag = model.encoder_diagnostics()
        assert diag is not None

    def test_stg_cae(self, data):
        x, _ = data
        model = StochasticGateConcreteModel(INPUT_DIM, K, sigma=0.5, total_epochs=EPOCHS)
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)

    def test_stg_ipcae(self, data):
        x, _ = data
        model = StochasticGateIndirectConcreteModel(
            INPUT_DIM, K, embedding_dim_encoder=ENCODER_EMB, total_epochs=EPOCHS
        )
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)

    def test_hcg_cae(self, data):
        x, _ = data
        model = HardConcreteGateConcreteModel(INPUT_DIM, K, total_epochs=EPOCHS)
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)

    def test_hcg_ipcae(self, data):
        x, _ = data
        model = HardConcreteGateIndirectConcreteModel(
            INPUT_DIM, K, embedding_dim_encoder=ENCODER_EMB, total_epochs=EPOCHS
        )
        model.train()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)
        model.eval()
        out = model(x)
        assert out.shape == (BATCH_SIZE, K)


class TestTrainingLoop:
    def test_encoder_training_step(self):
        model = ConcreteAutoencoderModel(INPUT_DIM, K, total_epochs=EPOCHS)
        classifier = _SimpleClassifier(K, 5)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(classifier.parameters()), lr=1e-3
        )
        x, y = _make_data()
        model.train()
        classifier.train()
        optimizer.zero_grad()
        features = model(x)
        output = classifier(features)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_combined_training_step(self):
        model = GumbelSoftmaxGateIndirectConcreteModel(
            INPUT_DIM, K, embedding_dim_encoder=ENCODER_EMB,
            embedding_dim_gate=GATE_EMB, total_epochs=EPOCHS
        )
        classifier = _SimpleClassifier(K, 5)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(classifier.parameters()), lr=1e-3
        )
        x, y = _make_data()
        model.train()
        classifier.train()
        optimizer.zero_grad()
        features = model(x)
        output = classifier(features)
        loss_cls = torch.nn.functional.cross_entropy(output, y)
        sparsity = model.sparsity_loss()
        loss = loss_cls + 1.0 * sparsity.total
        loss.backward()
        optimizer.step()
        assert loss.item() > 0
