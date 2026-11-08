import pytest
from unittest.mock import MagicMock, patch
from face_mosaic.io_utils import VideoWriter, VideoInfo
from face_mosaic.color import ColorMetadata

def _create_mock_process():
    proc = MagicMock()
    proc.poll.return_value = None
    proc.returncode = 0
    proc.stderr.read.return_value = b""
    return proc

def test_videowriter_encoder_cmd_branches():
    mock_input_info = MagicMock(spec=VideoInfo)
    mock_input_info.filepath = "input.mp4"
    mock_input_info.has_audio = True
    mock_input_info.color_metadata = ColorMetadata(
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_transfer="arib-std-b67",
        color_range="tv",
        pix_fmt="yuv420p10le"
    )

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=True), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=False), \
         patch("sys.platform", "darwin"):
        
        mock_popen.return_value = _create_mock_process()

        writer = VideoWriter(
            output_path="output_mac.mp4",
            input_info=mock_input_info,
            width=3840,
            height=2160,
            fps=60.0,
            codec="hevc",
            crf=18,
            use_hardware_accel=True
        )

        assert writer.active_encoder == "videotoolbox"
        called_cmd = mock_popen.call_args[0][0]
        assert "-c:v" in called_cmd
        enc_idx = called_cmd.index("-c:v")
        assert called_cmd[enc_idx + 1] == "hevc_videotoolbox"
        assert "-profile:v" in called_cmd
        assert "main10" in called_cmd
        writer.close()

def test_videowriter_qsv_cmd_branches():
    mock_input_info = MagicMock(spec=VideoInfo)
    mock_input_info.filepath = "input.mp4"
    mock_input_info.has_audio = False
    mock_input_info.color_metadata = None

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=False), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=True), \
         patch("sys.platform", "win32"):
        
        mock_popen.return_value = _create_mock_process()

        writer = VideoWriter(
            output_path="output_intel.mp4",
            input_info=mock_input_info,
            width=1920,
            height=1080,
            fps=30.0,
            codec="hevc",
            crf=18,
            use_hardware_accel=True
        )

        assert writer.active_encoder == "qsv"
        called_cmd = mock_popen.call_args[0][0]
        assert "hevc_qsv" in called_cmd
        writer.close()

def test_videowriter_cpu_threads_fallback():
    mock_input_info = MagicMock(spec=VideoInfo)
    mock_input_info.filepath = "input.mp4"
    mock_input_info.has_audio = False
    mock_input_info.color_metadata = None

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=False), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=False), \
         patch("face_mosaic.io_utils.supports_amf", return_value=False), \
         patch("face_mosaic.io_utils.supports_nvenc", return_value=False):
        
        mock_popen.return_value = _create_mock_process()

        writer = VideoWriter(
            output_path="output_cpu.mp4",
            input_info=mock_input_info,
            width=1920,
            height=1080,
            fps=30.0,
            codec="hevc",
            crf=18,
            use_hardware_accel=True
        )

        assert writer.active_encoder == "cpu"
        called_cmd = mock_popen.call_args[0][0]
        assert "libx265" in called_cmd
        assert "-threads" in called_cmd
        assert "0" in called_cmd
        assert "-movflags" in called_cmd
        assert "+faststart" in called_cmd
        writer.close()

def test_videowriter_faststart_and_bit_depth_8bit():
    mock_input_info = MagicMock(spec=VideoInfo)
    mock_input_info.filepath = "input.mov"
    mock_input_info.has_audio = False
    mock_input_info.color_metadata = ColorMetadata(
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_transfer="arib-std-b67",
        color_range="tv",
        pix_fmt="yuv420p10le"
    )

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=False), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=False), \
         patch("face_mosaic.io_utils.supports_amf", return_value=False), \
         patch("face_mosaic.io_utils.supports_nvenc", return_value=False):
        
        mock_popen.return_value = _create_mock_process()

        # Force 8-bit output even though input is 10-bit
        writer = VideoWriter(
            output_path="output_8bit.mp4",
            input_info=mock_input_info,
            width=3840,
            height=2160,
            fps=60.0,
            codec="hevc",
            crf=23,
            bit_depth="8bit"
        )

        called_cmd = mock_popen.call_args[0][0]
        assert "-movflags" in called_cmd
        assert "+faststart" in called_cmd
        
        # Check output pix_fmt (the one after encoder args)
        pix_indices = [i for i, x in enumerate(called_cmd) if x == "-pix_fmt"]
        assert len(pix_indices) >= 2
        # Second -pix_fmt is the output pix_fmt
        assert called_cmd[pix_indices[1] + 1] == "yuv420p"
        assert "-crf" in called_cmd
        crf_idx = called_cmd.index("-crf")
        assert called_cmd[crf_idx + 1] == "23"
        assert "-tag:v" in called_cmd
        tag_idx = called_cmd.index("-tag:v")
        assert called_cmd[tag_idx + 1] == "hvc1"
        writer.close()


def test_videowriter_auto_codec_matches_source_and_apple_tags():
    # Case 1: Source is HEVC .MOV -> auto resolves to HEVC with -tag:v hvc1
    info_hevc = MagicMock(spec=VideoInfo)
    info_hevc.filepath = "iphone_video.MOV"
    info_hevc.codec_name = "hevc"
    info_hevc.pix_fmt = "yuv420p10le"
    info_hevc.has_audio = True
    info_hevc.color_metadata = ColorMetadata(
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_transfer="arib-std-b67",
        color_range="tv",
        pix_fmt="yuv420p10le"
    )

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=False), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=True), \
         patch("sys.platform", "win32"):
        mock_popen.return_value = _create_mock_process()
        writer = VideoWriter(
            output_path="iphone_video_blurred.MOV",
            input_info=info_hevc,
            width=3840,
            height=2160,
            fps=60.0,
            codec="auto"
        )
        assert writer.resolved_codec == "hevc"
        cmd = mock_popen.call_args[0][0]
        assert "hevc_qsv" in cmd
        assert "-tag:v" in cmd
        assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
        assert "-profile:v" in cmd
        assert cmd[cmd.index("-profile:v") + 1] == "main10"
        assert "+cgop" in cmd
        writer.close()

    # Case 2: Source is H.264 .mp4 -> auto resolves to H.264 with -tag:v avc1
    info_h264 = MagicMock(spec=VideoInfo)
    info_h264.filepath = "camera_video.mp4"
    info_h264.codec_name = "h264"
    info_h264.pix_fmt = "yuv420p"
    info_h264.has_audio = False
    info_h264.color_metadata = None

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=False), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=True), \
         patch("sys.platform", "win32"):
        mock_popen.return_value = _create_mock_process()
        writer = VideoWriter(
            output_path="camera_video_blurred.mp4",
            input_info=info_h264,
            width=1920,
            height=1080,
            fps=30.0,
            codec="auto"
        )
        assert writer.resolved_codec == "h264"
        cmd = mock_popen.call_args[0][0]
        assert "h264_qsv" in cmd
        assert "-tag:v" in cmd
        assert cmd[cmd.index("-tag:v") + 1] == "avc1"
        assert "-profile:v" in cmd
        assert cmd[cmd.index("-profile:v") + 1] == "high"
        writer.close()


