import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { IResponse } from '../_interface/_response.interface';
@Injectable({
  providedIn: 'root'
})
export class PlotService {

    private plots: number[][];

    constructor() {
    }

    private http = inject(HttpClient);

    getPlots(): number[][]{
      return this.plots;
    }
    
    uploadFile(file: File): Observable<any> {
      const formData: FormData = new FormData();
      formData.append('file', file, file.name);
    
      return this.http.post('/api/plot/extract', formData).pipe(
        tap((response: IResponse) => {
          if (response.detail) {
            this.plots = response.detail;
          }
        })
      );
    }
    
  }

