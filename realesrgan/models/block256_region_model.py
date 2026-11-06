"""Whole-block network supervision with eligible local OCR polygons for CRNN loss."""
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.native_region_0921_model import NativeRegion0921Model


@MODEL_REGISTRY.register()
class Block256RegionModel(NativeRegion0921Model):
    @classmethod
    def _build_cp_annotations(cls, metadata, batch_size):
        """@luna-comment
        Input: producer-filtered local CP candidates in batch metadata. Output:nested polygon records.
        Key Steps: validate batch length and pass lists through; geometry/eligibility are not revalidated here.
        Key Params:no network crop/resize; grid_samples owns row filtering; polygon crops are auxiliary loss only.
        State/Dependencies:existing polygon CRNN loss and unchanged optimizer path.
        """
        lines=metadata.get('cp_lines',[[] for _ in range(batch_size)])
        if len(lines)!=batch_size:
            raise ValueError('Block CP annotation batch mismatch')
        return lines
