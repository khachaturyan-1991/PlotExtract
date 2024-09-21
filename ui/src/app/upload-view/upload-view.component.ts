import { Component, inject } from '@angular/core';
import { MessageService, PrimeNGConfig } from 'primeng/api';
import { PlotService } from '../_service/plot.service';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-upload-view',
  templateUrl: './upload-view.component.html',
  styleUrl: './upload-view.component.css'
})
export class UploadViewComponent {
    private messageService: MessageService = inject(MessageService);
    private translateService: TranslateService = inject(TranslateService);
    private plotService: PlotService = inject(PlotService);

    files = [];

    totalSize : number = 0;

    totalSizePercent : number = 0;

    maxSize : number = 1000000;

    choose(event, callback) {
        callback();
    }

    onRemoveTemplatingFile(event, file, removeFileCallback, index) {
        removeFileCallback(event, index);
        this.totalSize -= parseInt(this.formatSize(file.size));
        this.totalSizePercent = this.totalSize / 10;
    }

    
    onSelectedFiles(event) {
        this.files = event.currentFiles;
        this.totalSize = 0;
        this.files.forEach((file) => {
            this.totalSize += file.size;
        });
        this.totalSizePercent = (this.totalSize / this.maxSize) * 100;
    }

    uploadFiles() {
        if (this.files.length === 0) {
        this.messageService.add({ severity: 'warn', summary: 'NO_FILES_SELECTED', detail: 'PLEASE_SELECT_FILES_TO_UPLOAD', life: 3000 });
        return;
        }

        this.files.forEach(file => {
            this.plotService.uploadFile(file).subscribe({
                next: (response) => {
                    this.messageService.add({ severity: 'success', summary: this.translateService.instant('UPLOAD_SUCCESS'), detail: `${this.translateService.instant('FILE_UPLOADED_SUCCESSFULLY')}: ${file.name}`, life: 3000 });
                },
                error: (error) => {
                    this.messageService.add({ severity: 'error', summary: this.translateService.instant('UPLOAD_FAILED'), detail: `${this.translateService.instant('FAILED_TO_UPLOAD')} ${file.name}: ${error.message}`, life: 3000 });
                }
            });
        });
    }

    formatSize(bytes) {
        const k = 1024;
        const dm = 2;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }
}
