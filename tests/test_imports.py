
def test_imports():
    from src.models.transformer import TransformerEncoder
    from src.models.lstm import LSTMEncoder
    from src.models.fusion import AttentionFusion
    from src.models.autoencoder import AutoEncoder
    from src.models.ensemble import HybridAnomalyEnsemble
    assert TransformerEncoder and LSTMEncoder and AttentionFusion and AutoEncoder and HybridAnomalyEnsemble
