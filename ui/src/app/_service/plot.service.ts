import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { IResponse } from '../_interface/_response.interface';
@Injectable({
  providedIn: 'root'
})
export class PlotService {

    constructor() {
    }

    private http = inject(HttpClient);

    test(): Observable<IResponse> {
      return this.http.get<IResponse>('/api/plot/test');
    }
  }

