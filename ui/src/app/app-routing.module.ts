import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { UploadViewComponent } from './upload-view/upload-view.component';
import { ChartViewComponent } from './chart-view/chart-view.component';
import { ConfigViewComponent } from './config-view/config-view.component';

const routes: Routes = [
  {path: 'upload', component: UploadViewComponent},
  {path: 'chart', component: ChartViewComponent},
  {path: 'config', component: ConfigViewComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
