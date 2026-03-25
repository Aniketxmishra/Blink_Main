import pytest
import sys
from unittest.mock import patch
from blink.__main__ import main, serve

@patch('sys.argv', ['blink', '--version'])
@patch('builtins.print')
def test_main_version(mock_print):
    main()
    mock_print.assert_called()
    assert "blink-gpu" in mock_print.call_args_list[0][0][0]

@patch('sys.argv', ['blink'])
@patch('builtins.print')
def test_main_no_args(mock_print):
    # Depending on argparse setup, no args might print help and exit 0 or 2.
    # We catch SystemExit if argparse forces it.
    try:
        main()
    except SystemExit:
        pass
    assert mock_print.called

@patch('sys.argv', ['blink', 'predict', 'resnet18'])
@patch('blink._predictor.BlinkPredictor')
@patch('builtins.print')
def test_main_predict(mock_print, mock_predictor_class):
    mock_instance = mock_predictor_class.return_value
    mock_instance.predict.return_value = {
        'exec_time_ms': 100.0,
        'exec_time_lower': 90.0,
        'exec_time_upper': 110.0,
        'memory_mb': 500.0
    }
    main()
    mock_instance.predict.assert_called_once_with('resnet18', batch_size=32)
    mock_print.assert_called()

@patch('sys.argv', ['blink', 'dashboard'])
@patch('subprocess.run')
@patch('builtins.print')
def test_main_dashboard(mock_print, mock_run):
    main()
    mock_run.assert_called_once()
    assert 'streamlit' in mock_run.call_args[0][0]

@patch('sys.argv', ['blink', 'server'])
@patch('uvicorn.run')
@patch('builtins.print')
def test_main_server(mock_print, mock_run):
    main()
    mock_run.assert_called_once()

@patch('uvicorn.run')
@patch('builtins.print')
def test_serve_function(mock_print, mock_run):
    serve(host="127.0.0.1", port=9000, workers=2)
    mock_run.assert_called_once()
    assert mock_run.call_args[1]['host'] == "127.0.0.1"
    assert mock_run.call_args[1]['port'] == 9000
    assert mock_run.call_args[1]['workers'] == 2
