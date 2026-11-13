"""Render the saved experiment without retraining: python -m v2_engine.plot_content_benchmark."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    folder = Path(__file__).resolve().parents[2] / 'docs/content_benchmark'
    report = json.loads((folder / 'evaluation.json').read_text(encoding='utf-8'))
    history = pd.read_csv(folder / 'training_history.csv')
    figure = make_subplots(rows=1, cols=2, subplot_titles=(
        'Historical test MAE — lower is better', 'Candidate validation MAE during training'))
    for name, scores in report['test'].items():
        figure.add_trace(go.Bar(name=name.replace('_', ' '), x=[name.replace('_', ' ')],
                               y=[scores['mae']], text=[f"{scores['mae']:.2f}"], textposition='outside'), row=1, col=1)
    for (width, seed), rows in history.groupby(['width', 'seed']):
        figure.add_trace(go.Scatter(x=rows.epoch, y=rows.validation_mae, mode='lines',
                                   name=f'width {width}, seed {seed}'), row=1, col=2)
    figure.update_yaxes(title_text='MAE (score points)', row=1, col=1)
    figure.update_yaxes(title_text='MAE (score points)', row=1, col=2)
    figure.update_xaxes(title_text='Epoch', row=1, col=2)
    figure.update_layout(title='TrendSense: retrospective model comparison (not worldwide trend accuracy)',
                         template='plotly_white', height=600)
    figure.write_html(folder / 'comparison.html', include_plotlyjs=True)
    print(folder / 'comparison.html')


if __name__ == '__main__':
    main()
