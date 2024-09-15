import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HelpService {
  private http = inject(HttpClient);


  checkHelpExists(): Observable<{help: boolean}> {
    return this.http.get<{help: boolean}>('/api/helpexists');
  }
}
