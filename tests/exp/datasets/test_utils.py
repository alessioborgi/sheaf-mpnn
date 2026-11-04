import pytest
import torch
from src.exp.datasets.utils import RemoveSelfLoops, SubsetTrainSplit
from torch_geometric.data import Data


def test_remove_self_loops_misalignment():
    # Create a graph with 3 nodes and 4 edges: (0,0), (0,1), (1,1), (1,2)
    # Edges (0,0) and (1,1) are self-loops.
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.long)

    # Add edge_attr and edge_weight
    edge_attr = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])

    data = Data(
        edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight, num_nodes=3
    )

    transform = RemoveSelfLoops()
    transformed_data = transform(data)

    # After removing self-loops (0,0) and (1,1), we should have (0,1) and (1,2)
    expected_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    assert torch.equal(transformed_data.edge_index, expected_edge_index)

    # edge_weight should be [2.0, 4.0]
    expected_edge_weight = torch.tensor([2.0, 4.0])
    assert torch.equal(transformed_data.edge_weight, expected_edge_weight)

    # edge_attr should have 2 rows: [2.0, 2.0] and [4.0, 4.0]
    expected_edge_attr = torch.tensor([[2.0, 2.0], [4.0, 4.0]])

    assert transformed_data.edge_attr.size(0) == transformed_data.edge_index.size(1)
    assert torch.equal(transformed_data.edge_attr, expected_edge_attr)


def test_remove_self_loops_heterodata():
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # (u, e, u) has self-loops
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.long)
    data["u", "e", "u"].edge_index = edge_index
    data["u", "e", "u"].edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data["u"].num_nodes = 3

    # (u, e2, v) is bipartite, should be ignored
    edge_index2 = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    data["u", "e2", "v"].edge_index = edge_index2
    data["u", "e2", "v"].edge_weight = torch.tensor([10.0, 20.0])
    data["v"].num_nodes = 2

    transform = RemoveSelfLoops()
    transformed_data = transform(data)

    # Check (u, e, u)
    expected_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    assert torch.equal(transformed_data["u", "e", "u"].edge_index, expected_edge_index)
    assert torch.equal(
        transformed_data["u", "e", "u"].edge_weight, torch.tensor([2.0, 4.0])
    )

    # Check (u, e2, v) - should be unchanged
    assert torch.equal(transformed_data["u", "e2", "v"].edge_index, edge_index2)
    assert torch.equal(
        transformed_data["u", "e2", "v"].edge_weight, torch.tensor([10.0, 20.0])
    )


if __name__ == "__main__":
    test_remove_self_loops_misalignment()


# ---------------------------------------------------------------------------
# TestSubsetTrainSplit (formerly TrainValSplitter)
# ---------------------------------------------------------------------------


class TestSubsetTrainSplit:
    def test_bool_mask_creates_val_mask(self):
        """Bool train_mask is correctly split into train/val masks."""
        data = Data(
            x=torch.randn(10, 3),
            train_mask=torch.ones(10, dtype=torch.bool),
            num_nodes=10,
        )
        splitter = SubsetTrainSplit(val_ratio=0.2, seed=0)
        result = splitter(data)
        assert hasattr(result, "val_mask"), "val_mask should be created"
        assert hasattr(result, "train_mask"), "train_mask should still exist"
        assert result.val_mask.dtype == torch.bool
        assert result.train_mask.dtype == torch.bool
        # 20% of 10 = 2 validation nodes
        assert result.val_mask.sum().item() == 2
        assert result.train_mask.sum().item() == 8

    def test_integer_mask_treated_as_indices(self):
        """Non-bool train_mask (integer indices) is treated as an index list."""
        data = Data(
            x=torch.randn(10, 3),
            train_mask=torch.arange(10),  # integer indices, not bool
            num_nodes=10,
        )
        splitter = SubsetTrainSplit(val_ratio=0.2, seed=0)
        result = splitter(data)
        # val_mask should have floor(10 * 0.2) = 2 True entries
        assert result.val_mask.sum().item() == 2

    def test_use_index_true_stores_train_idx_val_idx(self):
        """use_index=True saves indices instead of masks."""
        data = Data(
            x=torch.randn(10, 3),
            train_mask=torch.ones(10, dtype=torch.bool),
            num_nodes=10,
        )
        splitter = SubsetTrainSplit(val_ratio=0.2, seed=0, use_index=True)
        result = splitter(data)
        assert hasattr(result, "train_idx"), (
            "train_idx should be set when use_index=True"
        )
        assert hasattr(result, "val_idx"), "val_idx should be set when use_index=True"
        assert result.train_idx.shape[0] == 8
        assert result.val_idx.shape[0] == 2

    def test_heterodata_path(self):
        """HeteroData: split is applied per node_type."""
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["paper"].x = torch.randn(10, 4)
        data["paper"].num_nodes = 10
        data["paper"].train_mask = torch.ones(10, dtype=torch.bool)
        splitter = SubsetTrainSplit(val_ratio=0.3, seed=0)
        result = splitter(data)
        assert hasattr(result["paper"], "val_mask"), "val_mask should be set on paper"
        assert result["paper"].val_mask.sum().item() == 3  # 30% of 10

    def test_no_train_mask_raises_value_error(self):
        """Data without train_mask should raise ValueError."""
        data = Data(x=torch.randn(5, 3), num_nodes=5)
        splitter = SubsetTrainSplit(val_ratio=0.2)
        with pytest.raises(ValueError, match="train_mask"):
            splitter(data)

    def test_non_bool_base_mask_rebuilt_correctly(self):
        """When original train_mask is non-bool, base mask is rebuilt from indices."""
        # 10 nodes, train_mask as indices 0..9
        data = Data(
            x=torch.randn(10, 3),
            train_mask=torch.arange(10),  # integer indices
            num_nodes=10,
        )
        splitter = SubsetTrainSplit(val_ratio=0.2, seed=1)
        result = splitter(data)
        # val_mask + train_mask should cover all 10 nodes (from index list 0..9)
        assert (result.train_mask | result.val_mask).all()

    def test_seed_reproducibility(self):
        """Same seed produces same split."""
        data1 = Data(
            train_mask=torch.ones(20, dtype=torch.bool),
            num_nodes=20,
        )
        data2 = Data(
            train_mask=torch.ones(20, dtype=torch.bool),
            num_nodes=20,
        )
        s1 = SubsetTrainSplit(val_ratio=0.2, seed=42)
        s2 = SubsetTrainSplit(val_ratio=0.2, seed=42)
        r1 = s1(data1)
        r2 = s2(data2)
        assert torch.equal(r1.val_mask, r2.val_mask)
