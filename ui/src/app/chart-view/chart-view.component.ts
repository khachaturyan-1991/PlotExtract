import { Component, effect, inject, OnInit, signal } from '@angular/core';
import { PlotService } from '../_service/plot.service';
import { MessageService } from 'primeng/api';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-chart-view',
  templateUrl: './chart-view.component.html',
  styleUrls: ['./chart-view.component.css']
})
export class ChartViewComponent implements OnInit {
    private messageService: MessageService = inject(MessageService);
    private plotService: PlotService = inject(PlotService);
    private route: ActivatedRoute = inject(ActivatedRoute);

    data: any;
    options: any;
    start = signal(0);
    end = signal(9);
    steps = signal(10);
    coefficientsArray: number[][];

    constructor(){       
        effect(() => {
            const labels = this.generateLabels(this.start(), this.end(), this.steps());
            this.generateGraph(labels);
        });

    }

    ngOnInit(){
        const labels = this.generateLabels(this.start(), this.end(), this.steps());
        this.coefficientsArray = this.plotService.getPlots();
        this.generateGraph(labels);
    }


    limitSteps(){
        if (this.steps()>100) {
            this.steps.set(100);
        }
    }

    roundNumber(number: number , nearest: number){
        return Math.round(number * nearest) / nearest;
    }

    generateLabels(start: number, end: number, numPoints: number): number[] {
        numPoints = (Math.min(numPoints ,100));
        const step = (end - start) / (numPoints - 1);
        return Array.from({ length: numPoints }, (_, i) => this.roundNumber((start + i * step), 100));
    }

    getDatasets(coefficientsArray: number[][], labels: number[]){
        return coefficientsArray.map((coefficients, index) => ({
            label: `Plot ${index + 1}`,
            data: this.generateYValues(coefficients, labels),
            fill: false,
            borderColor: this.getRandomColor(index), 
            tension: 0.4
        }));
    }

    generateYValues(coefficients: number[], xValues: number[]): number[] {
        return xValues.map(x => {
            // Calculate y = a*x^n + b*x^(n-1) + ... + c
            return coefficients.reduce((acc, coef, idx) => acc + coef * Math.pow(x, coefficients.length - 1 - idx), 0);
        });
    }

    getRandomColor(index: number): string {
        const colors = ['--blue-500', '--pink-500', '--green-500', '--orange-500', '--yellow-500'];
        const documentStyle = getComputedStyle(document.documentElement);
        return documentStyle.getPropertyValue(colors[index % colors.length]);
    }

    generateGraph(labels: number[]) {
        if (!this.coefficientsArray) return;

        const documentStyle = getComputedStyle(document.documentElement);
        const textColor = documentStyle.getPropertyValue('--text-color');
        const textColorSecondary = documentStyle.getPropertyValue('--text-color-secondary');
        const surfaceBorder = documentStyle.getPropertyValue('--surface-border');

        const datasets = this.getDatasets(this.coefficientsArray, labels);

        this.data = {
            labels,
            datasets
        };

        this.options = {
            maintainAspectRatio: false,
            aspectRatio: 0.6,
            plugins: {
                legend: {
                    labels: {
                        color: textColor
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: textColorSecondary
                    },
                    grid: {
                        color: surfaceBorder,
                        drawBorder: false
                    }
                },
                y: {
                    ticks: {
                        color: textColorSecondary
                    },
                    grid: {
                        color: surfaceBorder,
                        drawBorder: false
                    }
                }
            }
        };
    }
}
