from pathlib import Path
import tempfile
from face_mosaic.cli import collect_video_files

def test_collect_video_files_skips_appledouble_and_hidden():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create normal video file
        valid_video = tmp_path / "valid_test.MOV"
        valid_video.write_bytes(b"dummy")

        # Create Mac resource fork file
        appledouble_file = tmp_path / "._valid_test.MOV"
        appledouble_file.write_bytes(b"dummy_resource")

        # Create hidden file
        hidden_video = tmp_path / ".hidden_test.MOV"
        hidden_video.write_bytes(b"dummy_hidden")

        collected = collect_video_files([str(tmp_path)])
        names = [f.name for f in collected]

        assert "valid_test.MOV" in names
        assert "._valid_test.MOV" not in names
        assert ".hidden_test.MOV" not in names
        assert len(collected) == 1
