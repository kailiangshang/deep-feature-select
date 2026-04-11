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
from exp.trainers import EncoderTrainer, GateTrainer, GateEncoderTrainer, get_task_backend
from exp.utils import MLPClassifier, MLPRegressor, AutoencoderHead

INPUT_DIM = 100
K = 10
ENCODER_EMB = 16
GATE_EMB = 8
BATCH_SIZE = 4
EPOCHS = 3


class _FakeObs(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class _FakeAnnDataBatch:
    def __init__(self, x, y, target_key="cell_type"):
        self.X = x
        self.obs = _FakeObs({target_key: y})


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


class TestTaskBackends:
    def test_classification_backend(self):
        backend = get_task_backend("classification")
        output = torch.randn(4, 5, requires_grad=True)
        target = torch.randint(0, 5, (4,))
        loss = backend.compute_loss(output, target)
        assert loss.requires_grad
        preds = backend.predict(output)
        assert preds.shape == (4,)
        result = backend.evaluate(preds, target.numpy())
        assert "metric" in result
        assert backend.metric_name == "accuracy"

    def test_regression_backend(self):
        backend = get_task_backend("regression")
        output = torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3)
        loss = backend.compute_loss(output, target)
        assert loss.requires_grad
        preds = backend.predict(output)
        assert preds.shape == (4, 3)
        result = backend.evaluate(preds, target.numpy())
        assert "metric" in result
        assert backend.metric_name == "r2"

    def test_reconstruction_backend(self):
        backend = get_task_backend("reconstruction", input_dim=INPUT_DIM)
        output = torch.randn(4, INPUT_DIM, requires_grad=True)
        target = torch.randn(4, INPUT_DIM)
        loss = backend.compute_loss(output, target)
        assert loss.requires_grad
        preds = backend.predict(output)
        assert preds.shape == (4, INPUT_DIM)
        result = backend.evaluate(preds, target.numpy())
        assert "metric" in result
        assert backend.metric_name == "neg_mse"

    def test_encoder_trainer_classification(self):
        model = ConcreteAutoencoderModel(INPUT_DIM, K, total_epochs=3)
        head = MLPClassifier(K, 32, 5)
        trainer = EncoderTrainer(model, head, task="classification", lr=1e-3, seed=0)
        x = torch.randn(16, INPUT_DIM)
        y = torch.randint(0, 5, (16,))
        loader = [_FakeAnnDataBatch(x, y)]
        df = trainer.fit(loader, epochs=3)
        assert "accuracy" in df.columns
        assert "loss_task" in df.columns

    def test_encoder_trainer_reconstruction(self):
        model = ConcreteAutoencoderModel(INPUT_DIM, K, total_epochs=3)
        head = AutoencoderHead(K, 32, INPUT_DIM)
        trainer = EncoderTrainer(model, head, task="reconstruction",
                                 lr=1e-3, seed=0, input_dim=INPUT_DIM)
        x = torch.randn(16, INPUT_DIM)
        loader = [_FakeAnnDataBatch(x, x)]
        df = trainer.fit(loader, epochs=3)
        assert "neg_mse" in df.columns

    def test_gate_trainer_regression(self):
        model = StochasticGateModel(INPUT_DIM, sigma=0.5)
        head = MLPRegressor(INPUT_DIM, 32, 3)
        trainer = GateTrainer(model, head, task="regression",
                              sparse_loss_weight=1.0, lr=1e-3, seed=0)
        x = torch.randn(16, INPUT_DIM)
        y = torch.randn(16, 3)
        loader = [_FakeAnnDataBatch(x, y, target_key="target")]
        df = trainer.fit(loader, epochs=3)
        assert "r2" in df.columns
        assert "loss_sparsity" in df.columns

    def test_combined_trainer_classification(self):
        model = GumbelSoftmaxGateIndirectConcreteModel(
            INPUT_DIM, K, ENCODER_EMB, GATE_EMB, total_epochs=3
        )
        head = MLPClassifier(K, 32, 5)
        trainer = GateEncoderTrainer(model, head, task="classification",
                                     sparse_loss_weight=1.0, lr=1e-3, seed=0)
        x = torch.randn(16, INPUT_DIM)
        y = torch.randint(0, 5, (16,))
        loader = [_FakeAnnDataBatch(x, y)]
        result_df, feature_df = trainer.fit(loader, epochs=3)
        assert "accuracy" in result_df.columns
        assert feature_df.shape[1] == K
