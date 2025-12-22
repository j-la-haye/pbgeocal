"""
Tests for BINGO format parser.
"""

import pytest
import tempfile
from pathlib import Path

from gcp_reprojection.bingo_parser import (
    BINGOParser,
    BINGOObservation,
    BINGOGCPBlock,
    TimingFileParser,
    parse_bingo_file,
    parse_timing_file,
)


class TestBINGOParser:
    """Tests for BINGO correspondence file parser."""
    
    @pytest.fixture
    def sample_bingo_content(self):
        """Sample BINGO file content."""
        return """1 L4_UVT + 69_3_bldgs
1061  431.97  4.65812
1062  425.12  8.23451
-99
2 L4_UVT + 67_dmnd_bldg
1059  518.88  14.0513
-99
3 L4_UVT + 45_solar_panels
1043  179.254  -13.6656
1044  172.891  -9.87231
-99
"""
    
    @pytest.fixture
    def temp_bingo_file(self, sample_bingo_content):
        """Create a temporary BINGO file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bingo', delete=False) as f:
            f.write(sample_bingo_content)
            return f.name
    
    def test_parse_basic_file(self, temp_bingo_file):
        """Test parsing a basic BINGO file."""
        parser = BINGOParser()
        observations = parser.parse_file(temp_bingo_file)
        
        # Should have 5 observations total (2 + 1 + 2)
        assert len(observations) == 5
    
    def test_parse_gcp_ids(self, temp_bingo_file):
        """Test that GCP IDs are parsed correctly."""
        parser = BINGOParser()
        observations = parser.parse_file(temp_bingo_file)
        
        gcp_ids = set(obs.gcp_id for obs in observations)
        assert gcp_ids == {1, 2, 3}
    
    def test_parse_gcp_names(self, temp_bingo_file):
        """Test that GCP names are parsed correctly."""
        parser = BINGOParser()
        observations = parser.parse_file(temp_bingo_file)
        
        # Find observation for GCP 1
        gcp1_obs = [obs for obs in observations if obs.gcp_id == 1]
        assert len(gcp1_obs) == 2
        assert gcp1_obs[0].gcp_name == "L4_UVT + 69_3_bldgs"
    
    def test_parse_image_ids(self, temp_bingo_file):
        """Test that image IDs are parsed correctly."""
        parser = BINGOParser()
        observations = parser.parse_file(temp_bingo_file)
        
        image_ids = set(obs.image_id for obs in observations)
        assert image_ids == {1043, 1044, 1059, 1061, 1062}
    
    def test_parse_coordinates(self, temp_bingo_file):
        """Test that U, V coordinates are parsed correctly."""
        parser = BINGOParser()
        observations = parser.parse_file(temp_bingo_file)
        
        # Find specific observation
        obs = [o for o in observations if o.image_id == 1061][0]
        assert obs.u == pytest.approx(431.97)
        assert obs.v == pytest.approx(4.65812)
    
    def test_parse_negative_coordinates(self, temp_bingo_file):
        """Test parsing negative coordinates."""
        parser = BINGOParser()
        observations = parser.parse_file(temp_bingo_file)
        
        # Find observation with negative V
        obs = [o for o in observations if o.image_id == 1043][0]
        assert obs.v == pytest.approx(-13.6656)
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        parser = BINGOParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/path/file.bingo")
    
    def test_to_pixel_coordinates_v_up(self):
        """Test photo-to-pixel conversion with V axis up."""
        parser = BINGOParser(v_axis_up=True)
        
        # Image center should map to center
        u, v = parser.to_pixel_coordinates(0, 0, 1000, 800)
        assert u == pytest.approx(500.0)
        assert v == pytest.approx(400.0)
        
        # Top-right corner in photo-coords (positive U, positive V)
        # Should map to top-right in pixels (high U, low V)
        u, v = parser.to_pixel_coordinates(500, 400, 1000, 800)
        assert u == pytest.approx(1000.0)
        assert v == pytest.approx(0.0)
    
    def test_to_pixel_coordinates_v_down(self):
        """Test photo-to-pixel conversion with V axis down."""
        parser = BINGOParser(v_axis_up=False)
        
        # Image center should still map to center
        u, v = parser.to_pixel_coordinates(0, 0, 1000, 800)
        assert u == pytest.approx(500.0)
        assert v == pytest.approx(400.0)
        
        # With V down, positive V should give higher pixel V
        u, v = parser.to_pixel_coordinates(0, 100, 1000, 800)
        assert v == pytest.approx(500.0)


class TestTimingFileParser:
    """Tests for timing file parser."""
    
    @pytest.fixture
    def sample_timing_content_csv(self):
        """Sample CSV timing content."""
        return """image_id,time
1008,432001.123456
1009,432001.223456
1010,432001.323456
"""
    
    @pytest.fixture
    def sample_timing_content_space(self):
        """Sample space-delimited timing content."""
        return """1008 432001.123456
1009 432001.223456
1010 432001.323456
"""
    
    @pytest.fixture
    def temp_csv_timing_file(self, sample_timing_content_csv):
        """Create temporary CSV timing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_timing_content_csv)
            return f.name
    
    @pytest.fixture
    def temp_space_timing_file(self, sample_timing_content_space):
        """Create temporary space-delimited timing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_timing_content_space)
            return f.name
    
    def test_parse_csv_format(self, temp_csv_timing_file):
        """Test parsing CSV timing file."""
        parser = TimingFileParser()
        timings = parser.parse_file(temp_csv_timing_file)
        
        assert len(timings) == 3
        assert timings[1008] == pytest.approx(432001.123456)
        assert timings[1009] == pytest.approx(432001.223456)
    
    def test_parse_space_format(self, temp_space_timing_file):
        """Test parsing space-delimited timing file."""
        parser = TimingFileParser()
        timings = parser.parse_file(temp_space_timing_file)
        
        assert len(timings) == 3
        assert 1008 in timings
    
    def test_get_time_range(self, temp_csv_timing_file):
        """Test getting time range."""
        parser = TimingFileParser()
        timings = parser.parse_file(temp_csv_timing_file)
        
        min_time, max_time = parser.get_time_range(timings)
        assert min_time == pytest.approx(432001.123456)
        assert max_time == pytest.approx(432001.323456)
    
    def test_empty_time_range(self):
        """Test time range for empty timings."""
        parser = TimingFileParser()
        min_time, max_time = parser.get_time_range({})
        assert min_time == 0.0
        assert max_time == 0.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def temp_bingo_file(self):
        """Create temporary BINGO file."""
        content = """1 Test_GCP
1001  100.0  50.0
-99
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bingo', delete=False) as f:
            f.write(content)
            return f.name
    
    @pytest.fixture
    def temp_timing_file(self):
        """Create temporary timing file."""
        content = "1001,432001.0\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_parse_bingo_file(self, temp_bingo_file):
        """Test parse_bingo_file convenience function."""
        observations = parse_bingo_file(temp_bingo_file)
        assert len(observations) == 1
    
    def test_parse_timing_file(self, temp_timing_file):
        """Test parse_timing_file convenience function."""
        timings = parse_timing_file(temp_timing_file)
        assert 1001 in timings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
