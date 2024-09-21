import { Component, inject } from '@angular/core';
import {MenuItem, MessageService} from 'primeng/api';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-nav-menu',
  templateUrl: './nav-menu.component.html',
  styleUrls: ['./nav-menu.component.css']
})
export class NavMenuComponent {
  items: MenuItem[];

  activeItem: MenuItem;
  //private helpService: HelpService = inject(HelpService);
  private messageService: MessageService = inject(MessageService);
  private translateService: TranslateService = inject(TranslateService);

  ngOnInit() {
      this.loaditems();
      
      this.translateService.onLangChange.subscribe(() => {
        this.loaditems();
      });
      
      this.activeItem = this.items[1];
  }

  loaditems(){
    this.items = [
      {label: this.translateService.instant('UPLOAD'), icon: 'pi pi-upload', routerLink: ['/upload']},
      {label: this.translateService.instant('CHART'), icon: 'pi pi-chart-line', routerLink: ['/chart']}
    ];
  }

  openHelpPage(){
    this.messageService.add({severity: 'secondary', summary: this.translateService.instant('NO_INTEGRATED_HELP'), detail: this.translateService.instant('HELP_DOES_NOT_EXIST')});
    // this.helpService.checkHelpExists().subscribe(response => {
    //   if (response.help){
    //     window.open('/help/index.html', '_blank');
    //   }
    //   else{
    //    this.messageService.add({severity: 'info', summary: this.translateService.instant('NO_INTEGRATED_HELP'), detail: this.translateService.instant('HELP_DOES_NOT_EXIST')});
    //   }
    // })
  }
}
